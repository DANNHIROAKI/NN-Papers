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


def plot_imbalance(args: simutils.SimulationArgs,
                   p_list: np.ndarray,
                   res_list: List[dict]):
    task_name = args.task_name
    nops = simutils.task_meta[task_name]["nops"]
    agg_ids = simutils.task_meta[task_name]["agg_ids"]
    selected_qid = args.selected_qid
    x_values = (p_list * args.dsize).astype(int)

    nround_list = np.array([len(res['history']) for res in res_list])
    correct_list = np.array([is_same_float(res['xip_pred']['pred_value'], res['y_exact']) for res in res_list])
    qsamples_list = np.array([[qcfg['qsample'] for qcfg in res['history'][-1]['qcfgs']] for res in res_list])
    avg_samples_list = np.array([np.mean(qsamples[agg_ids]) for qsamples in qsamples_list])
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
        ax.set_xscale('log')
        ax.set_xticks(x_values)
        ax.set_xticklabels([f'{alpha}' for alpha in x_values], rotation=45)
        ax.grid()

    plt.tight_layout()
    # save the plot
    save_dir = os.path.join(args.save_dir, "imbalance")
    os.makedirs(save_dir, exist_ok=True)
    tag = args.get_tag()
    save_path = os.path.join(save_dir, f"imbalance_{tag}.png")
    plt.savefig(save_path)
    plt.savefig(os.path.join(save_dir, "imbalance.png"))
    print(f"Saved imbalance plot to {save_path}")


def plot_imbalance_simple(args: simutils.SimulationArgs,
                   p_list: np.ndarray,
                   res_list: List[dict]):
    task_name = args.task_name
    nops = simutils.task_meta[task_name]["nops"]
    agg_ids = simutils.task_meta[task_name]["agg_ids"]
    selected_qid = args.selected_qid
    # x_values = (p_list * args.dsize).astype(int)
    imbalance_ratio = (0.5 - p_list) / (0.5 + p_list)
    print(imbalance_ratio)
    x_values = np.arange(len(p_list))

    nround_list = np.array([len(res['history']) for res in res_list])
    correct_list = np.array([is_same_float(res['xip_pred']['pred_value'], res['y_exact']) for res in res_list])
    qsamples_list = np.array([[qcfg['qsample'] for qcfg in res['history'][-1]['qcfgs']] for res in res_list])
    non_agg_ids = np.setdiff1d(np.arange(nops), agg_ids)
    qsamples_list[:, non_agg_ids] = 0.0
    avg_samples_list = np.array([np.mean(qsamples[agg_ids]) for qsamples in qsamples_list])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()
    axes[0].plot(x_values, avg_samples_list)
    axes[0].set_title('Average Fraction of Samples')
    axes[0].set_ylabel('Average Fraction of Samples')
    for qid in range(nops):
        axes[1].plot(x_values, qsamples_list[:, qid], label=f'q{qid}')
    axes[1].set_title('Fraction of Samples of Each Feature')
    axes[1].set_ylabel('Fraction of Samples of Each Feature')

    for i, ax in enumerate(axes):
        ax.legend()
        ax.set_xlabel('Imbalance Ratio')
        # ax.set_xscale('log')
        ax.set_xticks(x_values)
        # ax.set_xticklabels([f'{alpha}' for alpha in x_values], rotation=45)
        ax.set_xticklabels([f'{alpha:.4g}' for alpha in imbalance_ratio], rotation=45)
        ax.grid()

    plt.tight_layout()
    # save the plot
    save_dir = os.path.join(args.save_dir, "imbalance_simple")
    os.makedirs(save_dir, exist_ok=True)
    tag = args.get_tag()
    save_path = os.path.join(save_dir, f"imbalance_{tag}.png")
    plt.savefig(save_path)
    plt.savefig(os.path.join(save_dir, "imbalance.png"))
    print(f"Saved imbalance plot to {save_path}")


if __name__ == "__main__":
    # p_list = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5])
    imbalance_ratio = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999])
    p_list = 0.5 * (1.0 - imbalance_ratio) / (1.0 + imbalance_ratio)
    p_list = np.maximum(p_list, 1e-5)
    res_list = []
    for p in tqdm(p_list, desc='p'):
        sim_args = simutils.SimulationArgs().parse_args()
        sim_args.dist_param = p
        res = simutils.run_default(sim_args)
        res_list.append(res)

    sim_args = simutils.SimulationArgs().parse_args()
    # plot_imbalance(sim_args, p_list, res_list)
    plot_imbalance_simple(sim_args, p_list, res_list)
