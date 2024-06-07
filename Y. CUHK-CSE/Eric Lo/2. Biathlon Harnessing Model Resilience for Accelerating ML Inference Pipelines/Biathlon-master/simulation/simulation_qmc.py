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


def plot_conftrace(
    sim_args: simutils.SimulationArgs,
    nsamples_list: np.ndarray,
    seeds_list: np.ndarray,
    res_list: List[dict],
):
    task_name = sim_args.task_name
    max_error = simutils.task_meta[task_name]["max_error"]
    mean_list = [res["pred_mean"] for res in res_list]
    std_list = [res["pred_std"] for res in res_list]
    conf_list = []
    for i, res in enumerate(res_list):
        qmc_preds = res["qmc_preds"]
        y_exact = res["y_exact"]
        conf = np.sum(np.abs(qmc_preds - y_exact) <= max_error) / len(qmc_preds)
        conf_list.append(conf)
    mean_list = np.array(mean_list).reshape(len(nsamples_list), len(seeds_list))
    std_list = np.array(std_list).reshape(len(nsamples_list), len(seeds_list))
    conf_list = np.array(conf_list).reshape(len(nsamples_list), len(seeds_list))

    # plot barplot for each nsamples
    # plot conf for each nsamples
    fig, ax = plt.subplots()
    for i, nsamples in enumerate(nsamples_list):
        ax.errorbar(
            nsamples,
            1.0 - np.mean(conf_list[i]),
            yerr=np.std(conf_list[i]),
            fmt="o",
            label=f"nsamples={nsamples}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("nsamples")
    ax.set_ylabel("Inference Uncertainty")
    ax.set_xticks(nsamples_list)
    ax.set_xticklabels([f"{alpha}" for alpha in nsamples_list], rotation=45)
    ax.legend()

    plt.tight_layout()
    tag = sim_args.get_tag()
    plt.savefig(os.path.join(sim_args.save_dir, "qmc", f"conftrace_{tag}.pdf"))
    print(
        f'conftrace saved to {os.path.join(sim_args.save_dir, "qmc", f"conftrace_{tag}.pdf")}'
    )
    plt.savefig("./qmc_conftrace.png")
    plt.show()

    fig, ax = plt.subplots()
    for i, nsamples in enumerate(nsamples_list):
        ax.errorbar(
            nsamples,
            np.mean(mean_list[i]),
            yerr=np.std(mean_list[i]),
            fmt="o",
            label=f"nsamples={nsamples}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("nsamples")
    ax.set_ylabel("prediction value")
    ax.set_xticks(nsamples_list)
    ax.set_xticklabels([f"{alpha}" for alpha in nsamples_list], rotation=45)
    ax.legend()

    plt.tight_layout()
    tag = sim_args.get_tag()
    plt.savefig(os.path.join(sim_args.save_dir, "qmc", f"predtrace_{tag}.pdf"))
    print(
        f'predtrace saved to {os.path.join(sim_args.save_dir, "qmc", f"predtrace_{tag}.pdf")}'
    )
    plt.savefig("./qmc_predtrace.png")


def plot_distribution(
    sim_args: simutils.SimulationArgs,
    nsamples_list: np.ndarray,
    seeds_list: np.ndarray,
    res_list: List[dict],
):
    task_name = sim_args.task_name
    max_error = simutils.task_meta[task_name]["max_error"]
    qmc_preds_list = [res["qmc_preds"] for res in res_list]

    seed = 1

    fig, ax = plt.subplots()
    for i, nsamples in enumerate(nsamples_list):
        preds_id = i * len(seeds_list) + seed
        qmc_preds = qmc_preds_list[preds_id]
        sns.kdeplot(qmc_preds, ax=ax, label=f"nsamples={nsamples}, seed={seed}")

    # add vertical line for exact value
    y_exact = res_list[0]["y_exact"]
    ax.axvline(y_exact, color="r", linestyle="--", label="exact value")

    ax.set_title(f"Inference Uncertanties of a request in {task_name}")
    ax.set_xlabel("prediction value")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    tag = sim_args.get_tag()
    plt.savefig(os.path.join(sim_args.save_dir, "qmc", f"distribution_{tag}.pdf"))
    print(
        f'distribution saved to {os.path.join(sim_args.save_dir, "qmc", f"distribution_{tag}.pdf")}'
    )
    plt.savefig("./qmc_distribution.png")

    fig, ax = plt.subplots()
    for i, nsamples in enumerate(nsamples_list):
        seed = 0
        preds_id = i * len(seeds_list) + seed
        qmc_preds = qmc_preds_list[preds_id]
        sns.histplot(
            qmc_preds, ax=ax, kde=True, label=f"nsamples={nsamples}, seed={seed}"
        )

    # add vertical line for exact value
    y_exact = res_list[0]["y_exact"]
    ax.axvline(y_exact, color="r", linestyle="--", label="exact value")

    ax.set_title(f"Inference Uncertanties of a request in {task_name}")
    ax.set_xlabel("prediction value")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    tag = sim_args.get_tag()
    plt.savefig(os.path.join(sim_args.save_dir, "qmc", f"histplot_{tag}.pdf"))
    print(
        f'histplot saved to {os.path.join(sim_args.save_dir, "qmc", f"histplot_{tag}.pdf")}'
    )
    plt.savefig("./qmc_histplot.png")


def single_req():
    res_list = []
    nsamples_list: np.ndarray = np.array([2**i for i in range(2, 12)])
    seeds_list: np.ndarray = np.arange(100)

    sim_args = simutils.SimulationArgs().parse_args()
    os.makedirs(os.path.join(sim_args.save_dir, "qmc"), exist_ok=True)
    tag = sim_args.get_tag()
    tag = "_".join(
        [
            f"{sim_args.get_tag()}",
            f"{nsamples_list[-1]}",
            f"{seeds_list[-1]}",
        ]
    )
    reslist_path = os.path.join(sim_args.save_dir, "qmc", f"reslist_{tag}.pkl")
    if not sim_args.nocache:
        if os.path.exists(reslist_path):
            print(f"Load results from {reslist_path}")
            res_list = joblib.load(reslist_path)

    if len(res_list) == 0:
        res_list = []
        for nsamples in tqdm(nsamples_list, desc="nsamples", leave=False):
            for seed in tqdm(seeds_list, desc="seed", leave=False):
                sim_args = simutils.SimulationArgs().parse_args()
                sim_args.pest_nsamples = nsamples
                sim_args.pest_seed = seed
                res = simutils.run_qmc(sim_args, verbose=False)
                res_list.append(res)
        joblib.dump(res_list, reslist_path)

    sim_args = simutils.SimulationArgs().parse_args()
    plot_conftrace(sim_args, nsamples_list, seeds_list, res_list)
    plot_distribution(sim_args, nsamples_list, seeds_list, res_list)


if __name__ == "__main__":
    single_req()