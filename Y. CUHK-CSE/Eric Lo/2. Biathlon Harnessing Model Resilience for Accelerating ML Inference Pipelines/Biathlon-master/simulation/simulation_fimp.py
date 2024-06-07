import os
import sys
import numpy as np
import json
import math
import time
from typing import List
from tap import Tap
from tqdm import tqdm
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox
import seaborn as sns
from sklearn import metrics
from scipy import stats
import warnings

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from apxinfer.core.config import OnlineArgs
from apxinfer.core.config import DIRHelper, LoadingHelper
from apxinfer.core.utils import is_same_float, XIPQType
from apxinfer.core.data import DBHelper
from apxinfer.core.pipeline import XIPPipeline
from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS
from apxinfer.examples.run import get_ppl

from apxinfer.simulation import utils as simutils


def plot_fimp(args: simutils.SimulationArgs,
                   rid_list: np.ndarray,
                   scale_list: np.ndarray,
                   res_list: List[dict]):
    task_name = args.task_name
    nops = simutils.task_meta[task_name]["nops"]
    agg_ids = simutils.task_meta[task_name]["agg_ids"]
    selected_qid = args.selected_qid
    x_values = np.arange(len(scale_list))

    nround_list = np.array([len(res['history']) for res in res_list])
    correct_list = np.array([is_same_float(res['xip_pred']['pred_value'], res['y_exact']) for res in res_list])
    qsamples_list = np.array([[qcfg['qsample'] for qcfg in res['history'][-1]['qcfgs']] for res in res_list])
    avg_samples_list = np.array([np.mean(qsamples[agg_ids]) for qsamples in qsamples_list])

    nround_list = np.mean(nround_list.reshape(len(rid_list), len(scale_list)), axis=0)
    correct_list = np.mean(correct_list.reshape(len(rid_list), len(scale_list)), axis=0)
    avg_samples_list = np.mean(avg_samples_list.reshape(len(rid_list), len(scale_list)), axis=0)
    qsamples_list = np.mean(qsamples_list.reshape(len(rid_list), len(scale_list), nops), axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    axes[0].plot(x_values, nround_list)
    axes[0].set_title('Number of Rounds')
    axes[0].set_ylabel('Number of Rounds')
    axes[1].plot(x_values, correct_list)
    axes[1].set_title('Correct Prediction')
    axes[1].set_ylabel('Correct Prediction')
    axes[2].plot(x_values, avg_samples_list)
    axes[2].set_title('Average Fraction of Samples')
    axes[2].set_ylabel('Average Fraction of Samples')
    for qid in range(nops):
        axes[3].plot(x_values, qsamples_list[:, qid], label=f'q{qid}')
    axes[3].set_title('Fraction of Samples')
    axes[3].set_ylabel('Fraction of Samples')

    for i, ax in enumerate(axes):
        ax.legend()
        ax.set_xlabel('imbalance degree')
        # ax.set_xscale('log')
        ax.set_xticks(x_values)
        ax.set_xticklabels([f'{scale}' for scale in scale_list], rotation=45)
        ax.grid()

    plt.tight_layout()
    # save the plot
    save_dir = os.path.join(args.save_dir, "fimp")
    os.makedirs(save_dir, exist_ok=True)
    tag = args.get_tag()
    save_path = os.path.join(save_dir, f"fimp_{tag}.png")
    plt.savefig(save_path)
    plt.savefig(os.path.join(save_dir, "fimp.png"))
    print(f"Saved fimp plot to {save_path}")


def run_vary_fimp(sim_args: simutils.SimulationArgs, scale: float, verbose: bool = False):
    assert sim_args.task_name == 'tickralfv2', 'Only support tickralfv2 task'

    if not sim_args.nocache:
        tag = sim_args.get_tag()
        os.makedirs(os.path.join(sim_args.save_dir, "results"), exist_ok=True)
        res_path = os.path.join(sim_args.save_dir, "results", f"res_{tag}_{scale}.pkl")
        if os.path.exists(res_path):
            if verbose:
                print(f"Load result from {res_path}")
            res = joblib.load(res_path)
            return res

    args: OnlineArgs = simutils.get_online_args(sim_args)
    test_set: pd.DataFrame = LoadingHelper.load_dataset(
        args, "test", args.nreqs, offset=args.nreqs_offset
    )

    ppl: XIPPipeline = get_ppl(sim_args.task_name, args, test_set, verbose=False)
    ppl.model.model.coef_[sim_args.selected_qid] *= scale
    if sim_args.debug:
        simutils.set_logger(logging.DEBUG, os.path.join(sim_args.save_dir, sim_args.logfile))

    request = test_set.iloc[sim_args.rid].to_dict()
    dsizes = [sim_args.dsize] * len(ppl.fextractor.queries)
    ddists = [sim_args.dist] * len(ppl.fextractor.queries)
    dist_args = [sim_args.dist_param] * len(ppl.fextractor.queries)
    operators = [
        [sim_args.op_type] if i == sim_args.selected_qid else None
        for i in range(len(ppl.fextractor.queries))
    ]

    res = simutils.run(
        ppl=ppl,
        request=request,
        dsizes=dsizes,
        ddists=ddists,
        dist_args=dist_args,
        operators=operators,
        syndb_seed=sim_args.syndb_seed,
        synv=sim_args.synv,
        keep_latency=sim_args.keep_latency,
    )
    if verbose:
        print(
            len(res["history"]),
            simutils.pred_check(sim_args.task_name, res["xip_pred"]["pred_value"], res["y_exact"]),
            res["latency"],
            [qcfg["qsample"] for qcfg in res["history"][-1]["qcfgs"]],
        )

    joblib.dump(res, res_path)
    if verbose:
        print(f"Saved result to {res_path}")

    return res


if __name__ == "__main__":
    sim_args = simutils.SimulationArgs().parse_args()
    assert sim_args.task_name == 'tickralfv2', 'Only support tickralfv2 task'

    rid_list = np.arange(1)
    scale_list = np.array([1e-4, 5e-4, 7e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    res_list = []
    for rid in tqdm(rid_list, desc='rid', leave=False):
        for scale in tqdm(scale_list, desc='scale', leave=False):
            sim_args = simutils.SimulationArgs().parse_args()
            sim_args.rid = rid
            res = run_vary_fimp(sim_args, scale=scale)
            res_list.append(res)

    sim_args = simutils.SimulationArgs().parse_args()
    plot_fimp(sim_args, rid_list, scale_list, res_list)
