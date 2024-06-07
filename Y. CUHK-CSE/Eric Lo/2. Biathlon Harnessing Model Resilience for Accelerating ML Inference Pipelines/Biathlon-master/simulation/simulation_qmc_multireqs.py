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
from apxinfer.core.utils import is_same_float, XIPQType, XIPFeatureVec
from apxinfer.core.data import DBHelper
from apxinfer.core.prediction import BiathlonPredictionEstimator
from apxinfer.core.pipeline import XIPPipeline
from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS
from apxinfer.examples.run import get_ppl

from apxinfer.simulation import utils as simutils


def plot_multireqs(
    sim_args: simutils.SimulationArgs, rid_list: np.ndarray, res_list: List[dict]
):
    task_name = sim_args.task_name
    fig, ax = plt.subplots()
    for i, rid in enumerate(rid_list):
        res = res_list[i]
        qmc_preds = res["qmc_preds"]
        y_exact = res["y_exact"]
        values = qmc_preds - y_exact
        sns.kdeplot(values, ax=ax, label=f"rid={rid}")

    # # add vertical line for exact value
    # y_exact = res_list[0]["y_exact"]
    # ax.axvline(y_exact, color="r", linestyle="--", label="exact value")

    ax.set_title(f"Inference Uncertanties of {len(rid_list)}x requests in {task_name}")
    ax.set_xlabel("prediction value")
    ax.set_ylabel("density")
    if len(rid_list) <= 10:
        ax.legend()
    plt.tight_layout()
    tag = sim_args.get_tag()
    plt.savefig(
        os.path.join(sim_args.save_dir, "qmc", f"distribution_multireqs_{tag}.pdf")
    )
    print(
        f'distribution saved to {os.path.join(sim_args.save_dir, "qmc", f"distribution_multireqs_{tag}.pdf")}'
    )
    plt.savefig("./qmc_distribution_multireqs.png")

    fig, ax = plt.subplots()
    for i, rid in enumerate(rid_list):
        res = res_list[i]
        qmc_preds = res["qmc_preds"]
        y_exact = res["y_exact"]
        values = qmc_preds - y_exact
        sns.histplot(values, ax=ax, kde=True, label=f"rid={rid}")

    # # add vertical line for exact value
    # y_exact = res_list[0]["y_exact"]
    # ax.axvline(y_exact, color="r", linestyle="--", label="exact value")

    ax.set_title(f"Inference Uncertanties of {len(rid_list)}x requests in {task_name}")
    ax.set_xlabel("prediction value")
    ax.set_ylabel("density")
    if len(rid_list) <= 10:
        ax.legend()
    plt.tight_layout()
    tag = sim_args.get_tag()
    plt.savefig(os.path.join(sim_args.save_dir, "qmc", f"histplot_multireqs_{tag}.pdf"))
    print(
        f'histplot saved to {os.path.join(sim_args.save_dir, "qmc", f"histplot_multireqs_{tag}.pdf")}'
    )
    plt.savefig("./qmc_histplot_multireqs.png")


def multi_reqs():
    res_list = []
    rid_list: np.ndarray = np.arange(5)

    sim_args = simutils.SimulationArgs().parse_args()
    os.makedirs(os.path.join(sim_args.save_dir, "qmc"), exist_ok=True)
    tag = sim_args.get_tag()
    tag = "_".join(
        [
            f"{sim_args.get_tag()}",
            f"rids{rid_list[-1]}",
        ]
    )
    reslist_path = os.path.join(sim_args.save_dir, "qmc", f"reslist_{tag}.pkl")
    if not sim_args.nocache:
        if os.path.exists(reslist_path):
            print(f"Load results from {reslist_path}")
            res_list = joblib.load(reslist_path)

    if len(res_list) == 0:
        res_list = []
        for rid in tqdm(rid_list, desc="request", leave=False):
            sim_args = simutils.SimulationArgs().parse_args()
            sim_args.rid = rid
            res = simutils.run_qmc(sim_args, verbose=False)
            res_list.append(res)
        joblib.dump(res_list, reslist_path)

    sim_args = simutils.SimulationArgs().parse_args()
    plot_multireqs(sim_args, rid_list, res_list)


if __name__ == "__main__":
    multi_reqs()
