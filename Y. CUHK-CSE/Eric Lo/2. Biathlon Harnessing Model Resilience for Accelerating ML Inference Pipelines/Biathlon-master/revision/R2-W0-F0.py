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
import gc

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


def get_tag(args: OnlineArgs, max_seed: int):
    task_home, task_name = args.task.split("/")
    tag = "_".join([task_name, f"{args.nreqs_offset}", f"{args.nreqs}", f"{max_seed}"])
    return tag


def run(
    args: OnlineArgs,
    seeds_list: np.ndarray,
    save_dir: str,
    logfile: str,
    nocache: bool = False,
) -> dict:
    task_home, task_name = args.task.split("/")
    assert task_home == "final"

    res_dir = os.path.join(save_dir, "results", "2.1.1")
    os.makedirs(res_dir, exist_ok=True)
    tag = get_tag(args, seeds_list[-1])
    res_path = os.path.join(res_dir, f"res_{tag}.pkl")
    if (not nocache) and os.path.exists(res_path):
        print(f"Load results from {res_path}")
        return joblib.load(res_path)

    test_set: pd.DataFrame = LoadingHelper.load_dataset(
        args, "test", args.nreqs, offset=args.nreqs_offset
    )
    req_cols = [col for col in test_set.columns if col.startswith("req_")]

    ppl: XIPPipeline = get_ppl(task_name, args, test_set, verbose=False)
    if args.verbose:
        log_dir = os.path.join(save_dir, "log")
        simutils.set_logger(logging.DEBUG, os.path.join(log_dir, logfile))

    online_dir = DIRHelper.get_online_dir(args)
    tag = ppl.settings.__str__()
    df_path = os.path.join(online_dir, f"final_df_{tag}.csv")
    assert os.path.exists(df_path), f"File not found: {df_path}"
    print(f"Loading {df_path}")
    df = pd.read_csv(df_path)
    df = df.iloc[: args.nreqs]

    fnames = ppl.fextractor.fnames
    noq = ppl.fextractor.num_queries
    qsample_cols = [f"qsamples_{i}" for i in range(noq)]

    fvecs_list = []
    oracle_fvec_list = []
    moments_list = []
    requests = test_set[req_cols].to_dict(orient="records")
    for rid, request in tqdm(
        enumerate(requests), total=len(requests), desc="Processing requests"
    ):
        qsamples = df.loc[rid, qsample_cols].to_numpy()
        qnparts = (qsamples * 100).astype(int)

        # run exact, make ncores = 0 and loading_mode=1 to be faster
        ppl.fextractor.ncores = 0
        for qry in ppl.fextractor.queries:
            qry.loading_mode = 1
            qry.set_enable_qcache()
            qry.set_enable_dcache()
            qry.profiles = []

        oracle_pred = ppl.serve(request=request, exact=True)
        oracle_fvec_list.append(oracle_pred["fvec"])

        rrdatas = []
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

        ppl.fextractor.ncores = 1
        for qry in ppl.fextractor.queries:
            qry.loading_mode = 0

        fvecs = []
        for sid, seed in enumerate(seeds_list):
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
                                qrng.binomial(1, qsamples[qid], total_n).astype(bool)
                            ]
                    else:
                        srrd = None
                    qry._dcache["cached_rrd"] = srrd
                    qry._dcache["cached_nparts"] = qnparts[qid]

                    qcfgs[qid]["qsample"] = qsamples[qid]
            fvec, qcosts = ppl.fextractor.extract(request, qcfgs)
            fvecs.append(fvec)
            # qtimes = np.array([qcost["time"] for qcost in qcosts])
            # print(f"{rid}-{sid}: {qtimes}")
            # print(f"{rid}-{sid}: {fvec['fvals']}")

        fvecs_list.append(fvecs)
        gc.collect()
    # return oracle_fvec_list, fvecs_list
    res = {
        "oracle_fvec_list": oracle_fvec_list,
        "fvecs_list": fvecs_list,
        "moments_list": moments_list,
    }
    joblib.dump(res, res_path)
    print(f"Save results to {res_path}")
    return res


def get_pvalues(
    args: OnlineArgs,
    res: dict,
    method: str = "shapiro",
) -> np.ndarray:
    oracle_fvec_list = res["oracle_fvec_list"]
    fvecs_list = res["fvecs_list"]

    nreqs = args.nreqs
    nof = fvecs_list[0][0]["fvals"].shape[0]

    oracle_fvals_list = np.array([fvec["fvals"] for fvec in oracle_fvec_list])
    p_values = np.zeros((nreqs, nof))
    for rid in range(nreqs):
        oracle_fvals = oracle_fvals_list[rid]
        fvecs = fvecs_list[rid]
        for fid in range(nof):
            fvals = np.array([fvec["fvals"][fid] for fvec in fvecs])
            errors = fvals - oracle_fvals[fid]
            if method == "shapiro":
                shapiro_test = stats.shapiro(errors)
            elif method == "normaltest":
                shapiro_test = stats.normaltest(errors)
            elif method == "kstest":
                shapiro_test = stats.kstest(errors, "norm")
            elif method == "jarque_bera":
                shapiro_test = stats.jarque_bera(errors)
            elif method == "kurtosistest":
                shapiro_test = stats.kurtosistest(errors)
            elif method == "skewtest":
                shapiro_test = stats.skewtest(errors)
            else:
                raise ValueError(f"Invalid method: {method}")
            pvalue = shapiro_test.pvalue
            p_values[rid, fid] = pvalue
    print(f"p_values shape: {p_values.shape}")

    return p_values


def plot_skewness(
    args: OnlineArgs,
    seeds_list: np.ndarray,
    res: dict,
    save_dir: str,
    method: str = "shapiro",
    significance_level: float = 0.05,
):
    task_home, task_name = args.task.split("/")
    nreqs = args.nreqs
    print(f"plottig skewness for {nreqs} requests in {task_name}")

    meta = simutils.task_meta[task_name]
    agg_qids = meta["agg_ids"]
    naggs = len(agg_qids)

    # pvalues = get_pvalues(args, res, method)
    moments_list = res["moments_list"]

    ncols = 4
    nrows = naggs // ncols + (naggs % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for i, qid in tqdm(enumerate(agg_qids), desc="Plotting Skewness"):
        ax = axes[i]
        q_moments = [moments for moments in moments_list if moments["qid"] == qid]
        qry_skewness = [moments["skew"][0] for moments in q_moments]
        qry_dsize = [moments["size"] for moments in q_moments]

        qry_skewness = np.array(qry_skewness)
        qry_skewness = qry_skewness[~np.isnan(qry_skewness)]

        sns.histplot(qry_skewness, ax=ax, kde=True, stat="density")
        ax.set_title(f"Query {qid} Skewness")
        ax.set_xlabel("Skewness")
        ax.set_ylabel("Density")

        print(f"query {qid} avg skewness: {np.mean(qry_skewness)}")
        print(f"query {qid} avg size: {np.mean(qry_dsize)}")

    plt.tight_layout()
    fig_dir = os.path.join(save_dir, "figs", "2.0.0")
    os.makedirs(fig_dir, exist_ok=True)
    tag = get_tag(args, seeds_list[-1])
    fig_path = os.path.join(fig_dir, f"data_skewness_{method}_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/data_skewness.png")
    print(f"Save figure to {fig_path}")


def print_moments(
    args: OnlineArgs,
    seeds_list: np.ndarray,
    res: dict,
    save_dir: str,
    method: str = "shapiro",
    significance_level: float = 0.05,
):
    task_home, task_name = args.task.split("/")
    nreqs = args.nreqs
    print(f"plottig skewness for {nreqs} requests in {task_name}")

    meta = simutils.task_meta[task_name]
    agg_qids = meta["agg_ids"]
    moments_list = res["moments_list"]

    for i, qid in enumerate(agg_qids):
        q_moments = [moments for moments in moments_list if moments["qid"] == qid]
        qry_dsize = [moments["size"] for moments in q_moments]
        qry_std = [moments["std"] for moments in q_moments]
        qry_skewness = [moments["skew"][0] for moments in q_moments]
        qry_kurtosis = [moments["kurtosis"][0] for moments in q_moments]

        qry_std = np.array([qstd if qstd is not None else 0.0 for qstd in qry_std])
        qry_skewness = np.array(
            [skew if skew is not None else 0.0 for skew in qry_skewness]
        )
        qry_kurtosis = np.array(
            [kurtosis if kurtosis is not None else 0.0 for kurtosis in qry_kurtosis]
        )
        qry_std = qry_std[~np.isnan(qry_std)]
        qry_skewness = qry_skewness[~np.isnan(qry_skewness)]
        qry_kurtosis = qry_kurtosis[~np.isnan(qry_kurtosis)]

        print(f"query {qid} : {np.mean(qry_dsize)}, {np.mean(qry_std)}, {np.mean(qry_skewness)}, {np.mean(qry_kurtosis)}")


class R2W1F1Args(Tap):
    nocache: bool = False
    save_dir: str = "/home/ckchang/ApproxInfer/revision/cache"
    logfile: str = "debug.log"
    method: str = "shapiro"


if __name__ == "__main__":
    exp_args = R2W1F1Args().parse_args(known_only=True)
    args = OnlineArgs().parse_args(known_only=True)
    seeds_list = np.arange(1000)
    res = run(
        args, seeds_list, exp_args.save_dir, exp_args.logfile, nocache=exp_args.nocache
    )

    plot_skewness(args, seeds_list, res, exp_args.save_dir, exp_args.method)
    print_moments(args, seeds_list, res, exp_args.save_dir, exp_args.method)

    # python R2-W0-F0.py --task final/machineryralfsimmedian0 --model mlp --scheduler_batch 8 --nreqs 338
    # python R2-W0-F0.py --task final/machineryralf --model mlp --scheduler_batch 8 --nreqs 338
    # python R2-W0-F0.py --task final/tripsralfv2 --model lgbm --max_error 1.5 --scheduler_batch 2 --nreqs 220
