import os
import sys
import numpy as np
import json
import math
import time
from typing import List, Tuple
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
from apxinfer.core.utils import is_same_float, XIPQType, XIPFeatureVec
from apxinfer.core.data import DBHelper
from apxinfer.core.prediction import BiathlonPredictionEstimator
from apxinfer.core.pipeline import XIPPipeline
from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS
from apxinfer.examples.run import get_ppl

from apxinfer.simulation import utils as simutils


def get_tag(args: OnlineArgs, ntasks: int):
    tag = "_".join(
        [
            f"{args.scheduler_init}",
            f"{args.scheduler_batch}",
            f"{args.min_conf:.4f}",
            f"{ntasks}",
        ]
    )
    return tag


def run(args: OnlineArgs, save_dir: str, logfile: str) -> dict:
    task_home, task_name = args.task.split("/")
    assert task_home == "final"

    # test_set: pd.DataFrame = LoadingHelper.load_dataset(
    #     args, "test", args.nreqs, offset=args.nreqs_offset
    # )

    # ppl: XIPPipeline = get_ppl(task_name, args, test_set, verbose=False)
    # if args.verbose:
    #     log_dir = os.path.join(save_dir, "log")
    #     simutils.set_logger(logging.DEBUG, os.path.join(log_dir, logfile))

    online_dir = DIRHelper.get_online_dir(args)
    tag = DIRHelper.get_eval_tag(args)
    # tag = ppl.settings.__str__()
    df_path = os.path.join(online_dir, f"final_df_{tag}.csv")
    assert os.path.exists(df_path), f"File not found: {df_path}"
    print(f"Loading {df_path}")
    df = pd.read_csv(df_path)
    nrounds = df["nrounds"]

    res = {"task_name": task_name, "nrounds": nrounds}

    return res


def plot_nrounds_distribution(res_list: List[dict], save_dir: str):
    print(f"plottig nrounds distribution for {len(res)} pipelines")

    task_home, task_name = args.task.split("/")
    ntasks = len(res_list)
    ncols = 7
    nrows = ntasks // ncols + (ntasks % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), sharey=False)
    axes = axes.flatten()

    for tid, task in enumerate(res_list):
        ax = axes[tid]
        nrounds = task["nrounds"]

        # sns.histplot(nrounds, ax=ax, bins=20, color="skyblue")
        # ax.set_yscale("log")
        # ax.set_ylabel("Log(Frequency)")

        sns.histplot(nrounds, ax=ax, bins=20, kde=False, color="skyblue", stat='probability')
        ax.set_xlabel("No. of Iterations")
        ax.set_ylabel("Percentage")
        ax.set_title(f"Task {task['task_name']}")

    plt.tight_layout()
    plt.subplots_adjust(wspace=.0)
    fig_dir = os.path.join(save_dir, "figs")
    tag = get_tag(args, ntasks)
    fig_path = os.path.join(fig_dir, f"nrounds_distribution_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/nrounds_distribution.png")
    print(f"Save figure to {fig_path}")


if __name__ == "__main__":
    args = OnlineArgs().parse_args()
    save_dir: str = "/home/ckchang/ApproxInfer/revision/cache"
    logfile: str = "debug.log"
    task_list = ['tripsralfv2', 'tickralfv2', 'batteryv2', 'turbofan',
                 'tdfraudralf2d', 'machineryralf', 'studentqnov2subset']
    res_list = []

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 20
    for task_name in task_list:
        meta = simutils.task_meta[task_name]
        args.task = f"final/{task_name}"
        args.model = meta["model"]
        args.max_error = meta["max_error"]
        args.scheduler_batch = meta["naggs"]
        res = run(args, save_dir, logfile)
        res_list.append(res)
    plot_nrounds_distribution(res_list, save_dir)

    all_nrounds = np.concatenate([res["nrounds"] for res in res_list])
    total_nrounds = all_nrounds.sum()
    print(f"Total number of rounds: {total_nrounds}")
    back_nrounds = (all_nrounds - 1).sum()
    failed_frequency = back_nrounds * 1.0 / total_nrounds
    print(f"Frequency of failed pipelines: {failed_frequency:.4f}")
