import os
import sys
import debugpy
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

rename_map = {
    "batteryv2": "Battery",
    "trubofan": "Turbofan",
    "machineryralf": "Bearing-Imbalance",
    "studentqno18": "Student-QA",
    # Added for R2-W4-F1
    "tripsralfv2": "Trip-Fare",
    "tripsralfv3": "Trip-Fare",
    "tickralfv2": "Tick-Price",
    "turbofan": "Turbofan",
    "tdfraudralf2d": "Fraud-Detection",
    "tdfraudralf2dv2": "Fraud-Detection",
    "studentqnov2subset": "Student-QA",
}


class R2W1F3Args(Tap):
    task_home: str = "final"

    tasks: List[str] = ["machineryralf", "machineryralfmedian"]

    seed: int = 0
    oracle_type: str = "exact"
    metric: str = "acc"

    save_dir: str = "/home/ckchang/ApproxInfer/revision/cache/results/2.1.3"
    fig_dir: str = "/home/ckchang/ApproxInfer/revision/cache/figs/2.1.3"
    nocache: bool = False

    auto_rid: bool = False
    srid: int = 0
    erid: int = 1
    nseeds: int = 100
    error_to: str = "real"
    collect_nunique: bool = False

    debug: bool = False
    phase: str = "e2e"  # collect_errors, final_errors


def collect_data(args: R2W1F3Args) -> pd.DataFrame:
    os.makedirs(args.save_dir, exist_ok=True)
    data = []
    for task_name in args.tasks:
        sim_args = simutils.SimulationArgs().from_dict(
            {
                "task_home": args.task_home,
                "task_name": task_name,
                "seed": args.seed,
                "bs_type": "descrete" if "median" in task_name else "fstd",
                "save_dir": args.save_dir,
            }
        )
        ol_args = simutils.get_online_args(sim_args)

        online_dir = DIRHelper.get_online_dir(ol_args)
        evals_tag = DIRHelper.get_eval_tag(ol_args)
        evals_path = os.path.join(online_dir, f"evals_{evals_tag}.json")

        with open(evals_path, "r") as f:
            evals = json.load(f)
        latency = evals["avg_ppl_time"]
        if args.oracle_type == "exact":
            accuracy = evals["evals_to_ext"][args.metric]
        else:
            accuracy = evals["evals_to_gt"][args.metric]

        ol_args.exact = True
        online_dir = DIRHelper.get_online_dir(ol_args)
        evals_tag = DIRHelper.get_eval_tag(ol_args)
        evals_path = os.path.join(online_dir, f"evals_{evals_tag}.json")
        if os.path.exists(evals_path):
            with open(evals_path, "r") as f:
                exact_evals = json.load(f)
            bsl_latency = exact_evals["avg_ppl_time"]
            if args.oracle_type == "exact":
                bsl_accuracy = exact_evals["evals_to_ext"][args.metric]
            else:
                bsl_accuracy = exact_evals["evals_to_gt"][args.metric]
            bsl_infos = {
                "bsl_latency": bsl_latency,
                "speedup": bsl_latency / latency,
                "bsl_accuracy": bsl_accuracy,
            }
        else:
            print(f"Exact evals not found for {task_name}")
            bsl_infos = {}

        data.append(
            {
                "task_name": task_name,
                "latency": latency,
                "accuracy": accuracy,
                "avg_nrounds": evals["avg_nrounds"],
                "avg_sample": evals["avg_sample"] / 8.0,
                **bsl_infos,
            }
        )
    df = pd.DataFrame(data)
    return df


def plot_data(args: R2W1F3Args, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 1, figsize=(4.5, 4))

    suffixed_task_names = [
        name.replace("simmedian", "+").replace("median", "*")
        .replace(name, rename_map.get(name, name))
        .replace(name[:-1], rename_map.get(name[:-1], name[:-1]))
        for name in df["task_name"]
    ]

    task_names = [
        name.replace(name[:-1], rename_map.get(name[:-1], name[:-1])) if name.endswith("*") or name.endswith("+")
        else name.replace(name, rename_map.get(name, name))
        for name in suffixed_task_names
    ]
    # for name in df["task_name"]:
    #     name = name.replace("simmedian", "+").replace("median", "*")
    #     name = name.replace(name[:-1], rename_map.get(name[:-1], name[:-1])).replace(name, rename_map.get(name, name))
    #     print()

    ax = axes  # compare latency of each task
    sns.barplot(x="task_name", y="latency", data=df, ax=ax)
    # sns.barplot(x="task_name", y="avg_sample", data=df, ax=ax)
    # sns.barplot(x="task_name", y="avg_nrounds", data=df, ax=ax)
    # annotate the value of y on top of the bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.02),
            ha="center",
            va="center",
            fontsize=20,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )
    ax.set_ylim([0, 2])
    ax.tick_params(axis='both', labelsize=20)
    ax.set_title(task_names[0], fontsize=20)
    ax.set_ylabel("Latency (s)", fontsize=20)
    ax.set_xlabel("",)
    ax.set_xticklabels(["Original", "Rep. by MED"], rotation=10)

    plt.tight_layout()
    fig_dir = args.fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    tag = "_".join([args.oracle_type, args.metric] + args.tasks)
    fig_path = os.path.join(fig_dir, f"avg_vs_median_latency_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/avg_vs_median.png")
    print(f"Save figure to {fig_path}")

    ax.clear()
    ax = axes  # compare accuracy of each task
    sns.barplot(x="task_name", y="accuracy", data=df, ax=ax)
    # annotate the value of y on top of the bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.02),
            ha="center",
            va="center",
            fontsize=20,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylim([0, 1.1])
    ax.set_title(task_names[0], fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.set_xlabel("",)
    ax.set_xticklabels(["Original", "Rep. by MED"], rotation=10)

    plt.tight_layout()
    fig_dir = args.fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    tag = "_".join([args.oracle_type, args.metric] + args.tasks)
    fig_path = os.path.join(fig_dir, f"avg_vs_median_accuracy_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/avg_vs_median.png")
    print(f"Save figure to {fig_path}")


def collect_error_distribution(args: R2W1F3Args) -> List[dict]:
    res_list = []
    # rng = np.random.default_rng(args.seed)
    rid_list = np.arange(args.srid, args.erid)
    seeds_list = np.arange(args.nseeds)
    for task_name in args.tasks:
        sim_args = simutils.SimulationArgs().from_dict(
            {
                "task_home": args.task_home,
                "task_name": task_name,
                "seed": args.seed,
                "bs_type": "descrete" if "median" in task_name else "fstd",
                "save_dir": args.save_dir,
            }
        )
        ol_args = simutils.get_online_args(sim_args)

        test_set: pd.DataFrame = LoadingHelper.load_dataset(
            ol_args, "test", ol_args.nreqs, offset=ol_args.nreqs_offset
        )
        req_cols = [col for col in test_set.columns if col.startswith("req_")]
        requests = test_set[req_cols].to_dict(orient="records")

        ppl: XIPPipeline = get_ppl(task_name, ol_args, test_set, verbose=False)
        fnames = ppl.fextractor.fnames

        median_fids = [fid for fid, name in enumerate(fnames) if "_median" in name]
        median_qids = []
        for qid, qry in enumerate(ppl.fextractor.queries):
            for fid, fname in enumerate(qry.fnames):
                if "median" in fname:
                    median_qids.append(qid)
                    break
        # median_qid = median_qids[0] if len(median_qids) > 0 else 0
        if len(median_fids) == 0:
            continue

        print(f"median_qids: {median_qids}, median_fids: {median_fids}")
        print(f"fnames: {fnames}")

        online_dir = DIRHelper.get_online_dir(ol_args)
        tag = ppl.settings.__str__()
        df_path = os.path.join(online_dir, f"final_df_{tag}.csv")
        assert os.path.exists(df_path), f"File not found: {df_path}"
        print(f"Loading {df_path}")
        df = pd.read_csv(df_path)
        # df = df.iloc[args.srid, args.erid]
        median_qsamples = df[[f"qsamples_{qid}" for qid in median_qids]]
        print(f"median_qsamples: {np.mean(median_qsamples > 0.05, axis=0)}")

        if args.auto_rid:
            # get row id with median_qsamples > 0.05
            rid_list = np.where(np.any(median_qsamples > 0.05, axis=1) > 0.5)[0]
            rid_list = rid_list[args.srid : args.erid]
            print(f"auto rid_list: {rid_list}")

        for rid in rid_list:
            request = requests[rid]
            rid_res_list = []
            rid_res_tag = "-".join([task_name, f"rid={rid}"])
            if args.error_to == "apx":
                rid_res_tag = "-".join([rid_res_tag, "apx"])
            rid_res_path = os.path.join(args.save_dir, f"red_res_{rid_res_tag}.pkl")
            if (not args.nocache) and os.path.exists(rid_res_path):
                print(f"Load red_res from {rid_res_path}")
                rid_res_list = joblib.load(rid_res_path)
                res_list.extend(rid_res_list)
                continue

            # run exact, make ncores = 0 and loading_mode=1 to be faster
            ppl.fextractor.ncores = 0
            for qry in ppl.fextractor.queries:
                qry.loading_mode = 1
                qry.set_enable_qcache()
                qry.set_enable_dcache()
                qry.profiles = []
                qry.festimator.err_module.bs_feature_correction = True
                qry.festimator.err_module.bs_type = "descrete"
            oracle_pred = ppl.serve(request=request, exact=True)

            rrdatas = []
            moments_list = []
            warnings.simplefilter("ignore", RuntimeWarning)
            for qid, qry in enumerate(ppl.fextractor.queries):
                if qry.qtype == XIPQType.AGG:
                    cached_rrd: np.ndarray = qry._dcache["cached_rrd"]
                    rrdatas.append(cached_rrd)
                    if cached_rrd is not None:
                        # print(f"{rid}-{qid}: {cached_rrd.shape}")
                        # check whether cached_rrd contains object dtype
                        if cached_rrd.dtype == np.dtype("O"):
                            # the data are string
                            # we can not compute moments for object dtype directly
                            # we take cached_rrd as descrete data, and compute
                            # the proportion of each unique value
                            unique, counts = np.unique(cached_rrd, return_counts=True)
                            moments_list.append(
                                {
                                    "rid": rid,
                                    "qid": qid,
                                    "size": len(cached_rrd),
                                    "mean": None,
                                    "std": None,
                                    "skew": None,
                                    "kurtosis": None,
                                    "unique": unique,
                                    "counts": counts,
                                }
                            )
                        else:
                            moments_list.append(
                                {
                                    "rid": rid,
                                    "qid": qid,
                                    "size": len(cached_rrd),
                                    "mean": np.mean(cached_rrd),
                                    "std": np.std(cached_rrd),
                                    "skew": stats.skew(cached_rrd),
                                    "kurtosis": stats.kurtosis(cached_rrd),
                                }
                            )
                    else:
                        moments_list.append(
                            {
                                "rid": rid,
                                "qid": qid,
                                "size": 0,
                                "mean": None,
                                "std": None,
                                "skew": None,
                                "kurtosis": None,
                            }
                        )
                else:
                    rrdatas.append(None)

            skip_rid = True
            for qid in median_qids:
                if rrdatas[qid] is not None:
                    if rrdatas[qid].shape[0] > 20:
                        if task_name == "tdfraudralf2dmedian":
                            # count disticnt value in rrdatas[qid]
                            unique = np.unique(rrdatas[qid])
                            if len(unique) > 1:
                                skip_rid = False
                                print(f"rid={rid} qid={qid} unique={unique}")
                                break
                        else:
                            skip_rid = False
                            break
            if skip_rid:
                print(
                    f"Skip rid={rid} {[rrdatas[qid].shape[0] if rrdatas[qid] is not None else None for qid in median_qids]}"
                )
                continue
            if args.debug:
                print(f"rrdatas: {rrdatas}")

            ppl.fextractor.ncores = 1
            for qry in ppl.fextractor.queries:
                qry.loading_mode = 0
                qry.set_enable_qcache()
                qry.set_enable_dcache()
                qry.profiles = []
            xip_pred = ppl.run_apx(request=request, keep_qmc=True)
            if args.debug:
                print(f"xip_pred: {xip_pred}")

            qcfgs = ppl.scheduler.get_latest_profile()["qcfgs"]
            qsamples = np.array([qcfg["qsample"] for qcfg in qcfgs])
            qnparts = np.round(qsamples * 100).astype(int)
            print(f"qnparts: {qnparts}")

            for qry in ppl.fextractor.queries:
                qry.festimator.err_module.bs_feature_correction = False
                qry.festimator.err_module.bs_type = "fstd"

            fvecs = []
            for sid, seed in tqdm(
                enumerate(seeds_list),
                total=len(seeds_list),
                desc=f"{task_name} rid={rid}",
            ):
                qcfgs = ppl.scheduler.get_final_qcfgs(request)
                for qid, qry in enumerate(ppl.fextractor.queries):
                    if qry.qtype == XIPQType.AGG:
                        qry.set_enable_qcache()
                        # qry.profiles = qry.profiles[-1:]
                        qry.profiles = []
                        qrng = np.random.default_rng(seed)
                        all_rrd: np.ndarray = rrdatas[qid]
                        if all_rrd is not None:
                            total_n = all_rrd.shape[0]
                            if is_same_float(qsamples[qid], 1.0):
                                srrd = all_rrd
                            else:
                                # get srrd using bernulli sampling to make sure
                                # we can estimate error of "count"
                                srrd = all_rrd[
                                    qrng.binomial(1, qsamples[qid], total_n).astype(
                                        bool
                                    )
                                ]
                        else:
                            srrd = None
                        qry._dcache["cached_req"] = request["req_id"]
                        qry._dcache["cached_nparts"] = qnparts[qid]
                        qry._dcache["cached_rrd"] = srrd
                        qcfgs[qid]["qsample"] = qsamples[qid]

                fvec, qcosts = ppl.fextractor.extract(request, qcfgs)
                fvecs.append(fvec)

            for qid, qry in enumerate(ppl.fextractor.queries):
                if qry.qtype == XIPQType.AGG:
                    for fname in qry.fnames:
                        if "_median" not in fname:
                            if not args.collect_nunique:
                                continue
                            elif "_unique" not in fname:
                                continue
                        fid = oracle_pred["fvec"]["fnames"].index(fname)
                        real_feature = oracle_pred["fvec"]["fvals"][fid]
                        real_pred = oracle_pred["pred_value"]
                        real_errors = [
                            fvec["fvals"][fid] - real_feature for fvec in fvecs
                        ]
                        apx_feature = xip_pred["fvec"]["fvals"][fid]
                        apx_pred = xip_pred["pred_value"]
                        apx_fests = xip_pred["fvec"]["fests"][fid]
                        if isinstance(apx_fests, (np.ndarray, list)):
                            if args.error_to == "real":
                                apx_errors = np.array(apx_fests) - real_feature
                            elif args.error_to == "apx":
                                apx_errors = np.array(apx_fests) - apx_feature
                            else:
                                raise ValueError(f"Unknown error_to: {args.error_to}")
                            rid_res_list.append(
                                {
                                    "task_name": task_name,
                                    "rid": rid,
                                    "fid": fid,
                                    "fname": fname,
                                    "real_feature": real_feature,
                                    "real_pred": real_pred,
                                    "real_errors": real_errors,
                                    "apx_feature": apx_feature,
                                    "apx_pred": apx_pred,
                                    "apx_errors": apx_errors,
                                }
                            )
                        else:
                            print(
                                f"Skip {task_name} rid={rid} qid={qid} fid={fid} fname={fname} {apx_fests}"
                            )
            joblib.dump(rid_res_list, rid_res_path)
            res_list.extend(rid_res_list)
    return res_list


def plot_error_distribution(args: R2W1F3Args, res_list: List[dict]):
    nres = len(res_list)
    ncols = 4
    nrows = (nres // ncols) + (nres % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, res in enumerate(res_list):
        ax = axes[i]
        task_name = res["task_name"]
        rid, fid = res["rid"], res["fid"]
        if not isinstance(res["apx_errors"], (np.ndarray, list)):
            print(f"Skip {task_name}-rid({rid})-f({fid})")
            continue
        if np.all(res["real_errors"] == res["real_errors"][0]):
            print(f"Zero Error at {task_name}-rid({rid})-f({fid})")

        sns.kdeplot(
            # [float(v) for v in json.loads(res["real_errors"])],
            res["real_errors"],
            label="Real Error",
            color="blue",
            alpha=0.5,
            ax=ax,
        )
        sns.kdeplot(
            # [float(v) for v in json.loads(res["apx_errors"])],
            res["apx_errors"],
            label="Bootstrap Error",
            color="red",
            alpha=0.5,
            ax=ax,
        )
        ax.legend()
        ax.set_xlabel("Error Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{task_name}")

    plt.tight_layout()
    fig_dir = args.fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    tag = "_".join([args.oracle_type, args.metric] + args.tasks)
    if args.error_to == "apx":
        tag = "_".join([tag, "apx"])
    fig_path = os.path.join(fig_dir, f"median_error_distribution_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/median_error_distribution.png")
    print(f"Save figure to {fig_path}")


def plot_final_error_dist(args: R2W1F3Args, res_list: List[dict]):
    suffixed_task_names = [
        res["task_name"].replace("simmedian", "+").replace("median", "")
        .replace(res["task_name"], rename_map.get(res["task_name"], res["task_name"]))
        .replace(res["task_name"][:-1], rename_map.get(res["task_name"][:-1], res["task_name"][:-1]))
        for res in res_list
    ]

    task_names = [
        name.replace(name[:-1], rename_map.get(name[:-1], name[:-1])) if name.endswith("*") or name.endswith("+")
        else name.replace(name, rename_map.get(name, name))
        for name in suffixed_task_names
    ]

    nres = len(res_list)
    # ncols = 4
    # nrows = (nres // ncols) + (nres % ncols > 0)
    ncols, nrows = 7, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()
    # sns.set(font_scale=1.1)
    for i, res in enumerate(res_list):
        ax = axes[i]
        ax.tick_params(axis='both', which='major', labelsize=14)
        task_name = task_names[i]
        rid, fid = res["rid"], res["fid"]
        if not isinstance(res["apx_errors"], (np.ndarray, list)):
            print(f"Skip {task_name}-rid({rid})-f({fid})")
            continue
        if np.all(res["real_errors"] == res["real_errors"][0]):
            print(f"Zero Error at {task_name}-rid({rid})-f({fid})")

        sns.kdeplot(
            res["real_errors"],
            label="Real Error",
            color="blue",
            alpha=0.5,
            ax=ax,
        )
        sns.kdeplot(
            res["apx_errors"],
            label="Bootstrap Error",
            color="red",
            alpha=0.5,
            ax=ax,
        )
        ax.legend()
        ax.set_xlabel("Error Value", fontsize=14)
        if i == 0:
            ax.set_ylabel("Density", fontsize=14)
        else:
            ax.set_ylabel("")
        ax.set_title(f"{task_name}", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=.3)
    fig_dir = args.fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    if args.error_to == "apx":
        fig_path = os.path.join(fig_dir, f"median_error_distribution_{args.error_to}.pdf")
    else:
        fig_path = os.path.join(fig_dir, f"median_error_distribution.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/median_error_distribution.png")
    print(f"Save figure to {fig_path}")


if __name__ == "__main__":
    args = R2W1F3Args().parse_args()
    if args.debug:
        try:
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            pass
    if args.phase == "final_errors":
        assert args.nocache == False, "Must use cache to plot final figure"
        selected = {
            # "tripsralfv2median": [(5, 5)],
            "tripsralfv3median": [(5, 5)],
            "tickralfv2median": [(4, 6)],
            "batteryv2median": [(11, 2)],
            "turbofanmedian": [(0, 1)],
            # "tdfraudralf2dmedian": [],  # not availabel
            "tdfraudralf2dv2median": [(5537, 6)],  # not ok yet
            "machineryralfmedian": [(3, 7)],
            "studentqnov2subsetmedian": [(18, 2)],
        }
        res_list = []
        for task_name, cfgs in selected.items():
            for rid, fid in cfgs:
                rid_res_tag = "-".join([task_name, f"rid={rid}"])
                if args.error_to == "apx":
                    rid_res_tag = "-".join([rid_res_tag, "apx"])
                rid_res_path = os.path.join(args.save_dir, f"red_res_{rid_res_tag}.pkl")
                if not os.path.exists(rid_res_path):
                    print(f"Skip {task_name} rid={rid} fid={fid}")
                    continue
                rid_res_list = joblib.load(rid_res_path)
                for res in rid_res_list:
                    if res["fid"] == fid:
                        res_list.append(res)
                        break
        plot_final_error_dist(args, res_list)
    elif args.phase == "e2e":
        df = collect_data(args)
        print(df)
        plot_data(args, df)
    elif args.phase == "collect_errors":
        res_list = collect_error_distribution(args)
        # print(res_list[:2])
        plot_error_distribution(args, res_list)
