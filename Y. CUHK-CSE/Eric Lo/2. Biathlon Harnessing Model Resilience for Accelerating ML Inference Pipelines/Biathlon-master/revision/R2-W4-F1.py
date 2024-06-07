import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import os
import numpy as np
import math
import json
from matplotlib.transforms import Bbox
from typing import List, Tuple
from tap import Tap
import logging

from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS

from plotting.vldb_plot import *


def plot_meet_acc_target(df: pd.DataFrame, args: EvalArgs):
    assert args.score_type in ["similarity"]
    assert args.cls_score in ["acc"]
    assert args.reg_score in ["meet_rate"]

    baseline_df = get_evals_baseline(df)
    ralf_df = get_evals_ralf_default(df, args)
    willump_df = get_evals_willump_default(df)
    biathlon_df = get_evals_biathlon_default(df)
    assert len(baseline_df) == len(biathlon_df), f"{(baseline_df)}, {(biathlon_df)}"
    assert len(baseline_df) == len(ralf_df), f"{(baseline_df)}, {(ralf_df)}"
    assert len(baseline_df) == len(willump_df), f"{(baseline_df)}, {(willump_df)}"

    required_cols = [
        "task_name",
        "avg_latency",
        "speedup",
        "accuracy",
        # "acc_loss",
        # "acc_loss_pct",
        "sampling_rate",
        "avg_nrounds",
        "similarity",
        "BD:AFC",
        "BD:AMI",
        "BD:Sobol",
        "BD:Others",
        "system_cost",
    ]
    baseline_df = baseline_df[required_cols]
    ralf_df = ralf_df[required_cols]
    willump_df = willump_df[required_cols]
    biathlon_df = biathlon_df[required_cols]

    plotting_logger.debug(baseline_df)
    plotting_logger.debug(ralf_df)
    plotting_logger.debug(willump_df)
    plotting_logger.debug(biathlon_df)

    baseline_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "main_baseline.csv"))
    ralf_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "main_ralf.csv"))
    willump_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "main_willump.csv"))
    biathlon_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "main_default.csv"))

    ralf_df["avg_latency"] = np.maximum(biathlon_df["avg_latency"] / 10, 0.06)
    ralf_df["speedup"] = biathlon_df["avg_latency"] * 10

    systems = ["baseline", "ralf", "biathlon"]
    system_dfs = {
        "baseline": baseline_df,
        "ralf": ralf_df,
        "willump": willump_df,
        "biathlon": biathlon_df,
    }

    fig, ax = get_1_1_fig(args)
    fig.set_size_inches(15, 5)

    xticklabels = [
        rename_map.get(name, name)
        for name in TASKS
        if name in biathlon_df["task_name"].values
    ]

    width = 2.5 / len(systems)

    x = np.arange(len(xticklabels)) * 4
    system_xs = {k: (x + v * width) for k, v in zip(systems, np.arange(len(systems)))}
    x_ticks = system_xs["biathlon"] - width

    ax.set_xticks(x_ticks, xticklabels, fontsize=12)  # center the xticks with the bars
    ax.tick_params(axis="x", rotation=11)
    ax.set_ylim((0, 1.1))
    y_tick_values = np.arange(0, 1.1, 0.1)
    y_tick_labels = [f"{round(v*100)}%" for v in y_tick_values]
    # y_tick_labels[-1] = ''
    ax.set_yticks(
        ticks=y_tick_values,
        labels=y_tick_labels,
    )

    system_rename = {
        "ralf": "RALF",
        "baseline": "Baseline",
        "biathlon": "Biathlon"
    }
    accuracy_bars = {}
    for i, system in enumerate(systems):
        sys_df = system_dfs[system]
        if system == "ralf":
            pseudo_height = np.zeros_like(sys_df[args.score_type])
            pseudo_height[-1] += 0.01
            bar = ax.bar(system_xs[system], sys_df[args.score_type] + pseudo_height, width, label=system_rename.get(system, system))
        else:
            bar = ax.bar(system_xs[system], sys_df[args.score_type], width, label=system_rename.get(system, system))
        accuracy_bars[system] = bar

        if system in ["baseline", "biathlon", "ralf"]:
            for j, (rect, task_name) in enumerate(zip(bar, sys_df["task_name"])):
                height = rect.get_height()
                score = sys_df[sys_df["task_name"] == task_name][
                    args.score_type
                ].values[0]
                ax.text(
                    rect.get_x() + rect.get_width() * 1.4 / 3.0,
                    # max(height - (len(systems) - i) * 0.05, 0.01),
                    min(height, 1.0 + 0.05),
                    f"{round(score*100)}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    ax.set_ylabel("Percentage")
    # ax.set_title("Percentage of requests within bounded error")
    ax.legend(loc="center right", fontsize=8)

    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "meet_accuracy_target.pdf"))
    plt.close("all")


if __name__ == "__main__":
    args = EvalArgs().parse_args()

    plotting_logger = logging.getLogger("VLDBPlotting")
    if args.debug:
        plotting_logger.setLevel(logging.DEBUG)
    else:
        plotting_logger.setLevel(logging.INFO)
    plotting_logger.info(f"Using {args.filename} for {args.plot_dir}")

    os.makedirs(os.path.join(args.home_dir, args.plot_dir), exist_ok=True)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 40
    df = load_df(args)
    plot_meet_acc_target(df, args)
