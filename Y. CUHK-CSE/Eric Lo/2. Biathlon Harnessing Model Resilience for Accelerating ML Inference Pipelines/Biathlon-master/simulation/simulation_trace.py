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


def plot_trace(args: simutils.SimulationArgs, res: dict):
    task_name = args.task_name
    nops = simutils.task_meta[task_name]["nops"]
    selected_qid = args.selected_qid

    pred_value_trace = np.array([hist["pred"]["pred_value"] for hist in res["history"]])
    pred_conf_trace = np.array([hist["pred"]["pred_conf"] for hist in res["history"]])
    pred_var_trace = np.array([hist["pred"]["pred_var"] for hist in res["history"]])
    qsamples_trace = np.array(
        [[qcfg["qsample"] for qcfg in hist["qcfgs"]] for hist in res["history"]]
    )

    fval_trace = [[] for i in range(len(res["history"]))]
    fstd_trace = [[] for i in range(len(res["history"]))]
    for i, hist in enumerate(res["history"]):
        fvals = hist["fvec"]["fvals"]
        fests = hist["fvec"]["fests"]
        fdists = hist["fvec"]["fdists"]
        for j in range(nops):
            fval_trace[i].append(fvals[j])
            if fdists[j] == "unknown":
                fstd_trace[i].append(np.std(fests[j]))
            elif fdists[j] == "fixed":
                fstd_trace[i].append(0)
            else:
                fstd_trace[i].append(fests[j])
    fval_trace = np.array(fval_trace)
    fstd_trace = np.array(fstd_trace)

    # plot the traces
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    x = np.arange(len(pred_value_trace))
    axes[0].plot(pred_conf_trace)
    axes[0].scatter(
        x,
        pred_conf_trace,
        c=[
            (
                "r"
                if simutils.pred_check(
                    task_name, pred_value_trace[i], res["y_exact"]
                )
                else "b"
            )
            for i in range(len(pred_value_trace))
        ],
    )
    axes[0].set_title("Prediction Confidence")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Confidence")

    axes[1].plot(pred_value_trace)
    axes[1].scatter(
        x,
        pred_value_trace,
        c=[
            (
                "r"
                if simutils.pred_check(
                    task_name, pred_value_trace[i], res["y_exact"]
                )
                else "b"
            )
            for i in range(len(pred_value_trace))
        ],
    )
    if task_name in ALL_REG_TASKS:
        # add a horizontal line for the true value
        axes[1].axhline(y=res["y_exact"], color="g", linestyle="--")
    axes[1].set_title("Prediction Value")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")

    axes[2].plot(fval_trace[:, selected_qid], label=f"f_{selected_qid}")
    axes[2].scatter(x, fval_trace[:, selected_qid])
    # add a horizontal line for the true value
    axes[2].axhline(y=res["feature"][selected_qid], color="g", linestyle="--")
    axes[2].set_title("Feature Value")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Feature Value")
    axes[2].legend()
    axes[2].grid()

    for i in range(nops):
        if i == selected_qid:
            continue
        axes[3].plot(fval_trace[:, i], label=f"f_{i}")
        axes[3].scatter(x, fval_trace[:, i])
    axes[3].set_title("Feature Value")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Feature Value")
    axes[3].legend()
    axes[3].grid()

    axes[4].plot(qsamples_trace[:, selected_qid], label=f"f_{selected_qid}")
    axes[4].scatter(x, qsamples_trace[:, selected_qid])
    axes[4].set_title("Query Sample Frac")
    axes[4].set_xlabel("Iteration")
    axes[4].set_ylabel("Query Sample Frac")
    # axes[4].set_yscale("log")
    axes[4].legend()
    axes[4].grid()

    for i in range(nops):
        if i == selected_qid:
            continue
        axes[5].plot(qsamples_trace[:, i], label=f"f_{i}")
        axes[5].scatter(x, qsamples_trace[:, i])
    axes[5].set_title("Query Sample Frac")
    axes[5].set_xlabel("Iteration")
    axes[5].set_ylabel("Query Sample Frac")
    # axes[5].set_yscale("log")
    axes[5].legend()
    axes[5].grid()

    plt.tight_layout()
    # save the plot
    save_dir = os.path.join(args.save_dir, "trace")
    os.makedirs(save_dir, exist_ok=True)
    tag = args.get_tag()
    save_path = os.path.join(save_dir, f"trace_{tag}.png")
    plt.savefig(save_path)
    plt.savefig(os.path.join(save_dir, "trace.png"))
    print(f"Saved trace plot to {save_path}")


if __name__ == "__main__":
    sim_args = simutils.SimulationArgs().parse_args()
    res = simutils.run_default(sim_args)
    plot_trace(sim_args, res)
