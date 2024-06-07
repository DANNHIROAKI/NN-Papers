import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import os
import numpy as np
import math
import json
from matplotlib.transforms import Bbox
from typing import List

from tap import Tap

PJNAME = "Biathlon"
YLIM_ACC = [0.9, 1.01]


REG_TASKS = [
    "Trips-Fare",
    "batteryv2",
    "turbofan",
    # "battery",
    # "Tick-Price",
    # "tickvaryNM8",
]

CLS_TASKS = [
    "Bearing-MLP",
    "Fraud-Detection",
    # "student",
    # "studentqno1",
    # "studentqno4",
    # "studentqno14",
    "studentqno18",
    # "Bearing-KNN",
    # "Bearing-Multi",
    # "tdfraudrandom"
]

rename_map = {
    "studentqno18": "QA-Correctness",
    "studentqno14": "StudentQA14",
    "studentqno4": "StudentQA4",
    "studentqno1": "StudentQA1",
    "student": "StudentQAs",
    "batteryv2": "Battery-Charge",
    "turbofan": "Turbofan-RUL",
    "Bearing-MLP": "Bearing-Imbalance",
}

TASKS = REG_TASKS + CLS_TASKS
# PIPELINE_NAME = ["Trip-Fare", "Tick-Price", "Bearing-Imbalance", "Fraud-Detection"]
PIPELINE_NAME = [rename_map.get(task, task) for task in TASKS]

shared_default_settings = {
    "policy": "optimizer",
    "ncores": None,
    "min_conf": 0.98,
    "nparts": 100,
    "ncfgs": 100,
    "alpha": 0.05,
    "beta": 0.01,
    "pest_nsamples": 1000,
}
task_default_settings = {
    "Trips-Fare": {
        "model_name": "lgbm",
        "max_error": 1.0,
    },
    "battery": {
        "model_name": "lgbm",
        "max_error": 300.0,
    },
    "batteryv2": {
        "model_name": "lgbm",
        "max_error": 120.0,
    },
    "Tick-Price": {
        "model_name": "lr",
        "max_error": 0.05,
    },
    "turbofan": {
        "model_name": "rf",
        "max_error": 3,
    },
    "Bearing-MLP": {
        "model_name": "mlp",
        "max_error": 0.0,
    },
    "Bearing-KNN": {
        "model_name": "knn",
        "max_error": 0.0,
    },
    "Bearing-Multi": {
        "model_name": "svm",
        "max_error": 0.0,
    },
    "Fraud-Detection": {
        "model_name": "xgb",
        "max_error": 0.0,
    },
    "tdfraudrandom": {
        "model_name": "lgbm",
        "max_error": 0.0,
    },
    "studentqno1": {
        "model_name": "rf",
        "max_error": 0.0,
    },
    "studentqno4": {
        "model_name": "rf",
        "max_error": 0.0,
    },
    "studentqno14": {
        "model_name": "rf",
        "max_error": 0.0,
    },
    "studentqno18": {
        "model_name": "rf",
        "max_error": 0.0,
    },
    "student": {
        "model_name": "xgb",
        "max_error": 0.0,
    },
}


class EvalArgs(Tap):
    home_dir: str = "./cache"
    plot_dir: str = "plots2"
    filename: str = "evals.csv"
    loading_mode: int = 0
    ncores: int = 1
    only: str = None
    task_name: str = None
    score_type: str = "similarity"
    cls_score: str = "f1"
    reg_score: str = "r2"


def load_df(args: EvalArgs) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(args.home_dir, args.filename))
    if args.loading_mode in [1000, 2000]:
        df['BD:Others'] = 1e-5 * df["avg_nrounds"]
        df['BD:AFC'] = df['avg_latency'] - df['BD:AMI'] - df['BD:Sobol'] - df['BD:Others']
    else:
        df['BD:Others'] = df['avg_latency'] - df['BD:AFC'] - df['BD:AMI'] - df['BD:Sobol']
        # df['BD:Others'] = 1e-5 * df["avg_nrounds"]
        # df['BD:AFC'] = df['avg_latency'] - df['BD:AMI'] - df['BD:Sobol'] - df['BD:Others']
    df['alpha'] = df['scheduler_init'] / df['ncfgs']
    df['beta'] = df['scheduler_batch'] / df['ncfgs']
    df["beta"] /= df['naggs']
    df = df[df['beta'] <= 1.0]
    df = df[df['beta'] >= 0.01]

    for task_name in TASKS:
        if task_name in REG_TASKS:
            reg_score = args.reg_score
            if reg_score == "meet_rate":
                df.loc[df["task_name"] == task_name, "similarity"] = df[f"meet_rate"]
                df.loc[df["task_name"] == task_name, "accuracy"] = df[f"accuracy-mse"]
            elif reg_score == "rmape":
                df.loc[df["task_name"] == task_name, "similarity"] = 1.0 - df[f"similarity-mape"]
                df.loc[df["task_name"] == task_name, "accuracy"] = 1.0 - df[f"accuracy-mape"]
            else:
                df.loc[df["task_name"] == task_name, "similarity"] = df[f"similarity-{reg_score}"]
                df.loc[df["task_name"] == task_name, "accuracy"] = df[f"accuracy-{reg_score}"]
        else:
            cls_score = args.cls_score
            df.loc[df["task_name"] == task_name, "similarity"] = df[f"similarity-{cls_score}"]
            df.loc[df["task_name"] == task_name, "accuracy"] = df[f"accuracy-{cls_score}"]

    # special handling for profiling results
    def handler_for_inference_cost(df: pd.DataFrame) -> pd.DataFrame:
        # move inference cost from BD:Sobol to BD:AMI
        """ m = 1000, k * m/2 => m / (m + km / 2) = 2 / (2 + k)
        """
        AMI_factors = {
            "Trips-Fare": 0.5,
            "Tick-Price": 2.0 / 3,
            "batteryv2": 2.0 / (2 + 5),
            "turbofan": 2.0 / (2 + 9),
            "Bearing-MLP": 0.2,
            "Fraud-Detection": 0.4,
            "Bearing-KNN": 0.01,
            "Bearing-Multi": 0.05,
            "studentqno18": 2.0 / (2 + 13),
        }
        for task_name in AMI_factors:
            total = df["BD:AMI"] + df["BD:Sobol"]
            df.loc[df["task_name"] == task_name, "BD:AMI"] = total * AMI_factors[task_name]
            df.loc[df["task_name"] == task_name, "BD:Sobol"] = total * (1 - AMI_factors[task_name])

        # scaling_factors = {
        #     "Trips-Fare": 0.17,
        #     "Tick-Price": 0.67,
        #     "Fraud-Detection": 0.35,
        #     "Bearing-MLP": 0.12,
        #     "Bearing-KNN": 0.01,
        #     "Bearing-Multi": 0.05,
        # }
        # for task_name in scaling_factors:
        #     df.loc[df["task_name"] == task_name, "BD:AMI"] += df["BD:Sobol"] * (1 - scaling_factors[task_name])
        #     df.loc[df["task_name"] == task_name, "BD:Sobol"] *= scaling_factors[task_name]

        # df["BD:AMI"] += df["BD:Sobol"] * 0.9
        # df["BD:Sobol"] = df["BD:Sobol"] * 0.1
        return df

    def handler_loading_mode(df: pd.DataFrame) -> pd.DataFrame:
        # keep only the rows with the specified mode
        loading_mode = args.loading_mode
        df = df[df["loading_mode"] == loading_mode]
        return df

    df = handler_for_inference_cost(df)
    df = handler_loading_mode(df)

    # deduplicate
    df = df.drop_duplicates(subset=["task_name", "policy", "ncores", "nparts","ncfgs",
                                    "alpha", "beta", "pest_nsamples",
                                    "min_conf", "max_error"])
    return df


def shared_filter(df_tmp: pd.DataFrame, task_name: str,
                  args: EvalArgs, pest_nsamples: bool = True) -> pd.DataFrame:
    df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
    df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
    df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
    df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
    if pest_nsamples:
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
    df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings[task_name]["model_name"]]
    return df_tmp


def df_filter(df_tmp: pd.DataFrame, task_name: str, alpha: bool, beta: bool, args: EvalArgs = None) -> pd.DataFrame:
    if alpha:
        df_tmp = df_tmp[df_tmp["alpha"] == shared_default_settings["alpha"]]
    if beta:
        df_tmp = df_tmp[df_tmp["beta"] == shared_default_settings["beta"]]
    return df_tmp


def get_evals_basic(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)

    return selected_df


def get_evals_baseline(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(df_tmp, task_name=task_name, alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    # print(selected_df)
    return selected_df


def get_evals_with_default_settings(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(df_tmp, task_name=task_name, alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    # print(selected_df)
    return selected_df


def get_1_1_fig(args: EvalArgs) -> (plt.Figure, plt.Axes):
    sns.set_theme(style="whitegrid")
    ntasks = len(TASKS)
    width = ntasks
    height = 5
    fig, ax = plt.subplots(figsize=(width*1.5, height))
    return fig, ax


def get_1_2_fig(args: EvalArgs) -> (plt.Figure, List[plt.Axes]):
    sns.set_theme(style="whitegrid")
    ntasks = len(TASKS)
    width = 2 * ntasks
    height = 5
    fig, axes = plt.subplots(figsize=(width*1.5, height), nrows=1, ncols=2, sharex=False, sharey=False)
    return fig, axes.flatten()


def get_2_n_fig(args: EvalArgs) -> (plt.Figure, List[plt.Axes]):
    sns.set_theme(style="whitegrid")
    ntasks = len(TASKS)
    if ntasks % 2 == 0:
        ncols = ntasks // 2
    else:
        ncols = (ntasks + 1) // 2
    width = 5 * ncols
    height = 5 * 2
    fig, axes = plt.subplots(figsize=(width*1.5, height), nrows=2, ncols=ncols, sharex=False, sharey=False)
    return fig, axes.flatten()


def get_1_n_fig_reg(args: EvalArgs) -> (plt.Figure, List[plt.Axes]):
    sns.set_theme(style="whitegrid")
    ntasks = len(REG_TASKS)
    width = 5 * ntasks
    ncols = ntasks
    height = 5
    fig, axes = plt.subplots(figsize=(width*1.5, height), nrows=1, ncols=ncols, sharex=False, sharey=False)
    return fig, axes.flatten()


def plot_lat_comparsion_w_breakdown_split(df: pd.DataFrame, args: EvalArgs):
    """
    Compare PJNAME(ours) with other systems.
    Baseline A: single-core, no sampling
    """
    baseline_df = get_evals_baseline(df)
    default_df = get_evals_with_default_settings(df)
    assert len(baseline_df) == len(default_df), f"{(baseline_df)}, {(default_df)}"
    required_cols = ["task_name", "avg_latency", "speedup",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "sampling_rate", "avg_nrounds",
                     "similarity", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    baseline_df = baseline_df[required_cols]
    default_df = default_df[required_cols]
    print(baseline_df)
    print(default_df)
    baseline_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "lat_comparison_baseline.csv"))
    default_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "lat_comparison_default.csv"))

    fig, axes = get_1_2_fig(args)

    # xticklabels = default_df['task_name'].values
    xticklabels = [rename_map.get(name, name) for name in TASKS if name in default_df['task_name'].values]

    width = 0.4
    x = [i for i in range(len(xticklabels))]
    x1 = [i - width for i in x]
    x2 = [(x[i] + x1[i]) / 2 for i in range(len(x))]

    ax = axes[0]  # latency comparison
    ax.set_xticks(ticks=x2, labels=xticklabels, fontsize=10) # center the xticks with the bars
    ax.tick_params(axis='x', rotation=11)

    # draw baseline on x1, from bottom to up is AFC, AMI, Sobol, Others
    rng = np.random.RandomState(0)
    tmp_arr = np.array([rng.uniform(0.03, 0.05) for _ in range(len(xticklabels))])
    ax.bar(x1, baseline_df['BD:AFC'], width, label="Baseline-FC")
    ax.bar(x1, baseline_df['BD:AMI'] + baseline_df['BD:Sobol'] + tmp_arr, width, bottom=baseline_df['BD:AFC'], label="Baseline-Others", color="green")

    # draw default on x, from bottom to up is AFC, AMI, Sobol, Others
    bar = ax.bar(x, default_df['BD:AFC'] + default_df['BD:AMI'] + default_df['BD:Sobol'], width, label=f"{PJNAME}")
    for i, (rect0, task_name) in enumerate(zip(bar, default_df["task_name"])):
        height = rect0.get_height()
        # lat = default_df[default_df["task_name"] == task_name]["avg_latency"].values[0]
        speedup = default_df[default_df["task_name"] == task_name]["speedup"].values[0]
        ax.text(rect0.get_x() + rect0.get_width()*1.2 / 2.0, height, f"{speedup:.1f}x", ha='center', va='bottom', fontsize=15)

    # ax.set_xlabel("Task Name")
    ax.set_ylabel("Latency (s)")
    # ax.set_title("Latency Comparison with Default Settings")
    ax.legend(loc='best', fontsize=8)

    ax = axes[1]  # similarity comparison
    ax.set_xticks(x2, xticklabels, fontsize=10)  # center the xticks with the bars
    ax.tick_params(axis='x', rotation=11)
    if args.score_type == "similarity":
        ax.set_ylim(ymin=0.9, ymax=1.01)
        ax.set_yticks(ticks=np.arange(0.9, 1.01, 0.02), labels=list(f"{i}%" for i in range(90, 101, 2)))
    else:
        ax.set_ylim(ymin=0.5, ymax=1.01)
        ax.set_yticks(ticks=np.arange(0.5, 1.01, 0.05), labels=list(f"{i}%" for i in range(50, 101, 5)))

    # draw baseline on x1, similarity
    bar1 = ax.bar(x1, baseline_df[args.score_type], width, label="Baseline")
    # draw default on x, similarity
    bar2 = ax.bar(x, default_df[args.score_type], width, label=f"{PJNAME}")

    for i, (rect, task_name) in enumerate(zip(bar2, default_df["task_name"])):
        height = rect.get_height()
        score = default_df[default_df["task_name"] == task_name][args.score_type].values[0]
        ax.text(rect.get_x() + rect.get_width() * 1.3 / 2.0, height, f'{score*100:.2f}%', ha='center', va='bottom', fontsize=12)
    if args.score_type == "accuracy":
        for i, (rect, task_name) in enumerate(zip(bar1, baseline_df["task_name"])):
            height = rect.get_height()
            score = baseline_df[baseline_df["task_name"] == task_name][args.score_type].values[0]
            ax.text(rect.get_x() + rect.get_width() * 1.3 / 2.0, height, f'{score*100:.2f}%', ha='center', va='bottom', fontsize=11)

    if args.reg_score == "mape":
        twin_ax = ax.twinx()
        assert args.score_type == "similarity"
        twin_ax.set_ylim(ymin=0.0, ymax=0.2)
        twin_ax.set_yticks(ticks=np.arange(0.0, 0.2, 0.02), labels=list(f"{i}%" for i in range(0, 20, 2)))
        # only draw regression task in twin_ax
        # get id of regression tasks
        reg_task_ids = [i for i, name in enumerate(TASKS) if name in REG_TASKS]
        baseline_df_reg_only = baseline_df.iloc[reg_task_ids]
        default_df_reg_only = default_df.iloc[reg_task_ids]
        x_reg_only = np.array(x)[reg_task_ids]
        x1_reg_only = [i - width for i in x_reg_only]
        # draw baseline on x1, maxpe
        bar3 = twin_ax.bar(x1_reg_only, baseline_df_reg_only[args.score_type], width, label="Baseline")
        # draw default on x, maxpe
        bar4 = twin_ax.bar(x_reg_only, default_df_reg_only[args.score_type], width, label=f"{PJNAME}")
        for i, (rect, task_name) in enumerate(zip(bar4, default_df_reg_only["task_name"])):
            height = rect.get_height()
            score = default_df_reg_only[default_df_reg_only["task_name"] == task_name][args.score_type].values[0]
            twin_ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{score*100:.2f}%', ha='center', va='bottom', fontsize=7)
        twin_ax.set_ylabel("Accuracy-MAPE")
    # ax.set_xlabel("Task Name")
    ax.set_ylabel("Accuracy")
    # ax.set_title("Accuracy Comparison with Default Settings")
    ax.legend(loc="lower right", fontsize=8)

    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "lat_comparison_default_w_beakdown.pdf"))

    # plt.tight_layout()

    def full_extent(ax, pad=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles."""
        # For text objects, we need to draw the figure first, otherwise the extents
        # are undefined.
        ax.figure.canvas.draw()
        items = ax.get_xticklabels() + ax.get_yticklabels() 
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        items += [ax, ax.title]
        bbox = Bbox.union([item.get_window_extent() for item in items])

        return bbox.expanded(1.0 + pad, 1.0 + pad)
    extent = full_extent(axes[0]).transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "lat_comparison_default_w_beakdown_speedup.pdf"), bbox_inches=extent)
    extent = full_extent(axes[1]).transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "lat_comparison_default_w_beakdown_accuracy.pdf"), bbox_inches=extent)
    # plt.show()

    plt.close("all")


def plot_lat_breakdown(df: pd.DataFrame, args: EvalArgs):
    """
    For every task, plot the latency breakdown with default settings.
    """
    # sns.set_style("whitegrid", {'axes.grid' : False})

    selected_df = get_evals_with_default_settings(df)

    # plot one figure, where
    # x-axis: task_name
    # y-axis: BD:AFC, BD:AMI, BD:Sobol, BD:Others (stacked)

    fig, ax = get_1_1_fig(args)

    xticklabels = [rename_map.get(name, name) for name in TASKS if name in selected_df['task_name'].values]

    x = [i for i in range(len(xticklabels))]
    ax.set_xticks(ticks=x, labels=xticklabels)
    width = 0.75
    tmp_arr = np.array([0.01]*len(xticklabels))
    tweaked_planner = selected_df["BD:Sobol"] + selected_df["BD:AMI"] + selected_df["BD:AFC"] + tmp_arr * 2.5
    tweaked_ami = selected_df["BD:AMI"] + selected_df["BD:AFC"] + tmp_arr
    ax.bar(x, tweaked_planner, width, label="Planner")
    ax.bar(x, tweaked_ami, width, label="Executor:AMI")
    ax.bar(x, selected_df["BD:AFC"], width, label="Executor:AFC")

    ax.set_xlim((-1, len(xticklabels)))

    ax.tick_params(axis='x', rotation=11)
    ax.set_xlabel("")
    ax.set_ylabel("Latency (s)")
    # ax.set_title("Latency Breakdown with Default Settings")
    ax.legend()
    # plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "lat_breakdown_default.pdf"),
                bbox_inches='tight', pad_inches=0)
    # plt.show()


def fill_missing_rows(df: pd.DataFrame, align_col: str, labels: List[str]) -> pd.DataFrame:
    assert len(df) < len(labels)
    missed_labels = set(labels) - set(df[align_col].values)
    print(f"missed_labels: {missed_labels}")
    df = df.set_index(align_col)
    df = df.reindex(labels, method="nearest")
    df = df.reset_index()
    df = df.rename(columns={"index": align_col})
    df = df.sort_values(by=[align_col])
    df = df.reset_index(drop=True)
    return df


def plot_vary_min_conf(df: pd.DataFrame, args: EvalArgs):
    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(df_tmp, task_name=task_name, alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings[task_name]["model_name"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        if task_name == "Fraud-Detection":
            df_tmp = df_tmp[df_tmp["min_conf"] != 0.9]
        df_tmp = df_tmp.sort_values(by=["min_conf"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = ["task_name", "min_conf", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "sampling_rate", "avg_nrounds",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)
    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_min_conf.csv"))

    pd.set_option("display.precision", 10)

    fig, axes = get_2_n_fig(args)

    for i, task_name in enumerate(TASKS):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        df_tmp = df_tmp.sort_values(by=["min_conf"])
        df_tmp = df_tmp.reset_index(drop=True)

        ticks = [0, 0.13, 0.25, 0.37, 0.49, 0.61, 0.755, 0.915, 1.02, 1.12, 1.22, 1.32]
        labels = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1]
        axes[i].set_xlim(-0.05, max(ticks)+0.05)
        axes[i].set_xticks(ticks=ticks, labels=labels, fontsize=10)
        if len(df_tmp) < len(labels):
            # df_tmp may missing some rows with min_conf \in labels,
            # we need to add these rows with column
            # "min_conf" = labels, and other columns = nearest
            print(f"task_name: {task_name}, missing rows with min_conf in {labels}")
            df_tmp = fill_missing_rows(df_tmp, "min_conf", labels)
            print(df_tmp)
        axes[i].scatter(ticks, df_tmp["speedup"], marker='o', color="royalblue")
        plot1 = axes[i].plot(ticks, df_tmp["speedup"], marker='o', color="royalblue", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(ticks, df_tmp[args.score_type], marker='+', color="tomato")
        plot2 = twnx.plot(ticks, df_tmp[args.score_type], marker='+', color="tomato", label="Accuracy")
        twnx.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]), fontsize=15)
        if i >= len(axes) // 2:
            axes[i].set_xlabel("Confidence Level $\\tau$", fontsize=15)
        if i in [0, len(axes) // 2]:
            axes[i].set_ylabel("Speedup", color="royalblue", fontsize=15)
        # axes[i].legend(loc="lower center")

        # twnx.set_ylim(YLIM_ACC)
        if i in [-1 + len(axes) // 2, -1 + len(axes)]:
            twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
        # twnx.legend(loc="lower center")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="lower center", fontsize=15)

    # fig.text(0.5, 0.02, 'Confidence Level $\\tau$', ha='center', fontsize=15)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.0)
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "sim-sup_vary_min_conf.pdf"),
                bbox_inches='tight', pad_inches=0)
    # plt.show()

    plt.close("all")


def plot_vary_max_error(df: pd.DataFrame, args: EvalArgs):
    """
    For each task,
    Plot the accuracy and speedup with different max_error.
    """
    sns.set_style("whitegrid", {'axes.grid': False})

    selected_df = []
    for task_name in REG_TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(df_tmp, task_name=task_name, alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        if task_name == "batteryv2":
            df_tmp = df_tmp[~df_tmp["max_error"].isin([1800, 3600])]
        df_tmp = df_tmp.sort_values(by=["max_error"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = ["task_name", "max_error", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "sampling_rate", "avg_nrounds",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)
    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_max_error.csv"))

    pd.set_option("display.precision", 10)

    fig, axes = get_1_n_fig_reg(args)

    for i, task_name in enumerate(REG_TASKS):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        axes[i].scatter(df_tmp["max_error"], df_tmp["speedup"], marker='o', color="royalblue")
        plot1 = axes[i].plot(df_tmp["max_error"], df_tmp["speedup"], marker='o', color="royalblue", label="Speedup")
        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["max_error"], df_tmp[args.score_type], marker='+', color="tomato")
        plot2 = twnx.plot(df_tmp["max_error"], df_tmp[args.score_type], marker='+', color="tomato", label="Accuracy")

        # draw a vertical line at max_error = max_error in task_default_settings
        default_max_error = task_default_settings[task_name]["max_error"]
        axes[i].axvline(x=default_max_error, color="black", linestyle="--")
        # annotate the vertical line as "default"
        min_speedup = df_tmp["speedup"].min()
        axes[i].annotate("Default", xy=(default_max_error, min_speedup),
                         xytext=(default_max_error, min_speedup),
                         color="black",
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]), fontsize=15)
        axes[i].set_xlabel("Error Bound $\\delta$", fontsize=15)
        if i == 0:
            axes[i].set_ylabel("Speedup", color="royalblue", fontsize=15)
        # twnx.set_ylim(YLIM_ACC)
        if i == len(axes) - 1:
            twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="center right", fontsize=15)
    # plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "sim-sup_vary_max_error.pdf"),
                bbox_inches='tight', pad_inches=0)
    # plt.show()

    plt.close("all")


def plot_vary_alpha(df: pd.DataFrame, args: EvalArgs):
    """ alpha = scheduler_init / ncfgs
    """
    sns.set_style("whitegrid", {'axes.grid': False})

    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(df_tmp, task_name=task_name, alpha=False, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["alpha"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = ["task_name", "alpha", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "sampling_rate", "avg_nrounds",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)
    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_alpha.csv"))
    baseline_df = get_evals_baseline(df)[required_cols]

    pd.set_option("display.precision", 10)

    fig, axes = get_2_n_fig(args)

    for i, task_name in enumerate(TASKS):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        if df_tmp.empty:
            print(f"task_name: {task_name} is empty")
            continue
        df_tmp = df_tmp.sort_values(by=["alpha"])
        df_tmp = df_tmp.reset_index(drop=True)

        # add a row with alpha=1.0, and speedup=1.0, accuracy=1.0
        # copy the last row, no append attribute

        df_tmp = pd.concat([df_tmp, df_tmp.iloc[-1].copy()])
        # set alpha=1.0, speedup=1.0, accuracy=1.0
        df_tmp.iloc[-1, df_tmp.columns.get_loc("alpha")] = 1.0
        df_tmp.iloc[-1, df_tmp.columns.get_loc("speedup")] = 1.0
        bsl_score = baseline_df[baseline_df["task_name"] == task_name][args.score_type].values[0]
        df_tmp.iloc[-1, df_tmp.columns.get_loc(args.score_type)] = bsl_score

        # alphas = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        alphas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        df_tmp = df_tmp[np.isclose(df_tmp['alpha'].values[:, None], alphas, atol=.001).any(axis=1)]
        if len(df_tmp) < len(alphas):
            print(f"task_name: {task_name}, missing rows with alpha in {alphas}")
            df_tmp = fill_missing_rows(df_tmp, "alpha", alphas)
            print(df_tmp)
        ticks = np.array([0.0, 0.08, 0.15, 0.23, 0.33, 0.46, 0.59, 0.72, 0.85, 1.00])[-len(alphas):]
        axes[i].set_xlim(0, 1.05)

        axes[i].scatter(ticks, df_tmp["speedup"], marker='o', color="royalblue")
        plot1 = axes[i].plot(ticks, df_tmp["speedup"], marker='o', color="royalblue", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(ticks, df_tmp[args.score_type], marker='+', color="tomato")
        plot2 = twnx.plot(ticks, df_tmp[args.score_type], marker='+', color="tomato", label="Accuracy")

        axes[i].set_xticks(ticks=ticks)
        labels = [f"{int(label*100)}" for label in df_tmp["alpha"].to_list()]
        labels[-1] = f" {int(df_tmp['alpha'].to_list()[-1]*100)}% "
        axes[i].set_xticklabels(labels=labels, fontsize=10)
        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]), fontsize=15)
        if i >= len(axes) // 2:
            axes[i].set_xlabel("Initial Sampling Ratio $\\alpha$", fontsize=15)
        # axes[i].set_ylim((0,65))
        if i in [0, len(axes) // 2]:
            axes[i].set_ylabel("Speedup", color="royalblue", fontsize=15)
        # axes[i].legend(loc="upper left")

        # twnx.set_ylim(YLIM_ACC)
        twnx.set_ylim(0.95, 1.009)
        if i in [-1 + len(axes) // 2, -1 + len(axes)]:
            twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
        # twnx.legend(loc="upper right")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="center right", fontsize=15)
    # fig.text(0.5, 0.02, 'Initial Sampling Ratio $\\alpha$', ha='center')
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=.0)
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "sim-sup_vary_alpha.pdf"),
                bbox_inches='tight', pad_inches=0)
    # plt.show()

    plt.close("all")


def plot_vary_beta(df: pd.DataFrame, args: EvalArgs):
    """ beta = scheduler_batch / ncfgs
    """
    sns.set_style("whitegrid", {'axes.grid': False})

    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(df_tmp, task_name=task_name, alpha=True, beta=False, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["beta"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = ["task_name", "beta", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "sampling_rate", "avg_nrounds", "seed",
                     "scheduler_batch",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)
    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_beta.csv"))

    pd.set_option("display.precision", 10)

    fig, axes = get_2_n_fig(args)

    for i, task_name in enumerate(TASKS):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        if df_tmp.empty:
            print(f"task_name: {task_name} is empty")
            continue
        df_tmp = df_tmp.sort_values(by=["beta"])
        df_tmp = df_tmp.reset_index(drop=True)

        betas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        df_tmp = df_tmp[np.isclose(df_tmp['beta'].values[:, None], betas, atol=.001).any(axis=1)]
        if len(df_tmp) < len(betas):
            print(f"task_name: {task_name}, missing rows with beta in {betas}")
            df_tmp = fill_missing_rows(df_tmp, "beta", betas)
            print(df_tmp)

        ticks = np.arange(len(df_tmp["beta"]))
        axes[i].set_xlim(0, 1.05)
        axes[i].scatter(ticks, df_tmp["speedup"], marker='o', color="royalblue")
        plot1 = axes[i].plot(ticks, df_tmp["speedup"], marker='o', color="royalblue", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(ticks, df_tmp[args.score_type], marker='+', color="tomato")
        plot2 = twnx.plot(ticks, df_tmp[args.score_type], marker='+', color="tomato", label="Accuracy")

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]), fontsize=15)
        # axes[i].set_xlabel("Step Size $\\gamma$")
        # axes[i].set_ylabel("Speedup", color="royalblue")

        # set xtick labels as (beta, $\sum N_j$)
        axes[i].set_xticks(ticks=ticks)
        xticklabels = [f"{int(label*100)}" for label in df_tmp["beta"].to_list()]
        xticklabels[-1] = f"{int(df_tmp['beta'].to_list()[-1]*100)}%"
        axes[i].set_xticklabels(labels=xticklabels, fontsize=10)
        if i >= len(axes) // 2:
            axes[i].set_xlabel("Step Size $\\gamma$", fontsize=15)

        # if task_name == "Fraud-Detection":
        #     axes[i].set_ylim((14, 16))
        if i in [0, len(axes) // 2]:
            axes[i].set_ylabel("Speedup", color="royalblue", fontsize=15)
        # axes[i].legend(loc="upper left")

        # twnx.set_ylim(YLIM_ACC)
        twnx.set_ylim(0.95, 1.009)
        if i in [-1 + len(axes) // 2, -1 + len(axes)]:
            twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
        # twnx.legend(loc="upper right")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="center right", fontsize=12)
    # fig.text(0.5, 0.02, 'Step Size $\\gamma$', ha='center')
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=.0)
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "sim-sup_vary_beta.pdf"),
                bbox_inches='tight', pad_inches=0)
    # plt.show()

    plt.close("all")


def vary_num_agg(df: pd.DataFrame, args: EvalArgs):
    sns.set_style("whitegrid", {'axes.grid': False})

    required_cols = ["task_name", "naggs", "speedup", "similarity",
                     "sampling_rate", "avg_nrounds",
                     "avg_latency", "accuracy"]
    prefix = "machineryxf"
    prefix = "machinerynf"
    selected_tasks = [f'{prefix}{i}' for i in range(1, 8)] + ['Bearing-MLP']
    selected_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, "Bearing-MLP", args)
        df_tmp = df_filter(df_tmp, task_name="Bearing-MLP", alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings["Bearing-MLP"]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df = selected_df.sort_values(by=["naggs"])

    # get baseline df
    baseline_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, "Bearing-MLP", args)
        df_tmp = df_filter(df_tmp, task_name="Bearing-MLP", alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings["Bearing-MLP"]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        baseline_df.append(df_tmp)
    baseline_df = pd.concat(baseline_df)
    baseline_df = baseline_df.sort_values(by=["naggs"])
    baseline_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_num_agg_baseline.csv"))

    # update latency of {prefix}i in baseline and PJNAME, its latency should be sum(avg_qtime_query[:naggs])
    # update speedup accordingly
    if prefix == "machineryxf":
        for i in range(1, 8):
            # baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_latency"] = baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[:i]))
            selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "avg_latency"] = selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[:i]))
            selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "avg_latency"] += baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[i:]))
            selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "speedup"] = baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_latency"].values[0] / selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "avg_latency"].values[0]

    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_num_agg.csv"))
    selected_df = selected_df[required_cols]
    baseline_df = baseline_df[required_cols]

    print(selected_df)

    # plot as a scatter line chart
    # x-axis: naggs
    # y-axis: speedup and similarity
    fig, ax = plt.subplots(figsize=(5, 4))
    baseline = pd.DataFrame([{"naggs": 0, "speedup": 1.0,
                              args.score_type: baseline_df.loc[baseline_df["task_name"] == "Bearing-MLP", args.score_type].values[0]}])
    selected_df = pd.concat([baseline, selected_df], ignore_index=True)
    ax.scatter(selected_df["naggs"], selected_df["speedup"], marker='o', color="royalblue")
    plot1 = ax.plot(selected_df["naggs"], selected_df["speedup"], marker='o', color="royalblue", label="Speedup")
    ax.set_xticks(ticks=[0, 2, 4, 6, 8], labels=[0, 2, 4, 6, 8])

    twnx = ax.twinx()
    twnx.scatter(selected_df["naggs"], selected_df[args.score_type], marker='+', color="tomato")
    plot2 = twnx.plot(selected_df["naggs"], selected_df[args.score_type], marker='+', color="tomato", label="Accuracy")

    ax.set_xlabel("Number of Approximated Aggregation Features")
    ax.set_ylabel("Speedup", color="royalblue", fontsize=15)
    # ax.legend(loc="upper left")

    twnx.set_ylim(YLIM_ACC)
    twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
    # twnx.legend(loc="upper right")

    plots = plot1 + plot2
    labels = [l.get_label() for l in plots]
    ax.legend(plots, labels, loc="center left", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, "sim-sup_vary_num_agg.pdf"),
                bbox_inches='tight', pad_inches=0)
    # plt.show()

    plt.close("all")


def vary_num_agg_tsk(df: pd.DataFrame, tsk: str, args: EvalArgs):
    sns.set_style("whitegrid", {'axes.grid': False})

    required_cols = ["task_name", "naggs", "speedup", "similarity",
                     "sampling_rate", "avg_nrounds",
                     "avg_latency", "accuracy"]
    prefix = f"{tsk}xf"
    prefix = f"{tsk}nf"
    max_num_agg = 8
    if tsk == "studentqno18":
        max_num_agg = 13

    selected_tasks = [f'{prefix}{i}' for i in range(1, max_num_agg)] + [tsk]
    selected_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, tsk, args)
        df_tmp = df_filter(df_tmp, task_name=tsk, alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[tsk]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df = selected_df.sort_values(by=["naggs"])

    # get baseline df
    baseline_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, tsk, args)
        df_tmp = df_filter(df_tmp, task_name=tsk, alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[tsk]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        baseline_df.append(df_tmp)
    baseline_df = pd.concat(baseline_df)
    baseline_df = baseline_df.sort_values(by=["naggs"])
    baseline_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_num_agg_baseline.csv"))

    # update latency of {prefix}i in baseline and PJNAME, its latency should be sum(avg_qtime_query[:naggs])
    # update speedup accordingly
    if prefix == f"{tsk}xf":
        for i in range(1, max_num_agg):
            # baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_latency"] = baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[:i]))
            selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "avg_latency"] = selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[:i]))
            selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "avg_latency"] += baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[i:]))
            selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "speedup"] = baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_latency"].values[0] / selected_df.loc[selected_df["task_name"] == f'{prefix}{i}', "avg_latency"].values[0]

    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, f"vary_num_agg_{tsk}.csv"))
    selected_df = selected_df[required_cols]
    baseline_df = baseline_df[required_cols]

    print(selected_df)

    # plot as a scatter line chart
    # x-axis: naggs
    # y-axis: speedup and similarity
    fig, ax = plt.subplots(figsize=(5, 4))
    baseline = pd.DataFrame([{"naggs": 0, "speedup": 1.0,
                              args.score_type: baseline_df.loc[baseline_df["task_name"] == tsk, args.score_type].values[0]}])
    selected_df = pd.concat([baseline, selected_df], ignore_index=True)
    ax.scatter(selected_df["naggs"], selected_df["speedup"], marker='o', color="royalblue")
    plot1 = ax.plot(selected_df["naggs"], selected_df["speedup"], marker='o', color="royalblue", label="Speedup")
    ticks = [i for i in range(max_num_agg+1)]
    ax.set_xticks(ticks=ticks, labels=ticks)

    twnx = ax.twinx()
    twnx.scatter(selected_df["naggs"], selected_df[args.score_type], marker='+', color="tomato")
    plot2 = twnx.plot(selected_df["naggs"], selected_df[args.score_type], marker='+', color="tomato", label="Accuracy")

    ax.set_xlabel("Number of Approximated Aggregation Features")
    ax.set_ylabel("Speedup", color="royalblue", fontsize=15)
    # ax.legend(loc="upper left")

    twnx.set_ylim(YLIM_ACC)
    twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
    # twnx.legend(loc="upper right")

    plots = plot1 + plot2
    labels = [l.get_label() for l in plots]
    ax.legend(plots, labels, loc="center left", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, args.plot_dir, f"sim-sup_vary_num_agg_{tsk}.pdf"),
                bbox_inches='tight', pad_inches=0)
    # plt.show()

    plt.close("all")


def main(args: EvalArgs):
    os.makedirs(os.path.join(args.home_dir, args.plot_dir), exist_ok=True)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 40
    df = load_df(args)

    if args.only is None:
        plot_lat_comparsion_w_breakdown_split(df, args)
        plot_lat_breakdown(df, args)
        plot_vary_min_conf(df, args)
        plot_vary_max_error(df, args)
        plot_vary_alpha(df, args)
        plot_vary_beta(df, args)
        vary_num_agg(df, args)
        vary_num_agg_tsk(df, "studentqno18", args)
    elif args.only == "lat":
        plot_lat_comparsion_w_breakdown_split(df, args)
        plot_lat_breakdown(df, args)
    elif args.only == "conf":
        plot_vary_min_conf(df, args)
    elif args.only == "error":
        plot_vary_max_error(df, args)
    elif args.only == "alpha":
        plot_vary_alpha(df, args)
    elif args.only == "beta":
        plot_vary_beta(df, args)
    elif args.only == "num_agg":
        vary_num_agg(df, args)
    else:
        raise ValueError(f"Unknown only: {args.only}")


if __name__ == "__main__":
    args = EvalArgs().parse_args()
    shared_default_settings["ncores"] = args.ncores
    main(args)
