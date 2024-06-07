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
import statsmodels.api as sm

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


class R2W1F2Args(Tap):
    ds_type: str = "bimodal"
    dsize: int = 10_000_000 + 1
    # dsize: int = 1_000_000 + 1
    # dsize: int = 100_000 + 1
    sfrac: float = 0.05
    nseeds: int = 1000
    bs_seed: int = 0
    nresamples: int = 100

    nocache: bool = False
    save_dir: str = "/home/ckchang/ApproxInfer/revision/cache/results/2.1.2"

    def process_args(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def get_tag(self):
        tag = "-".join(
            [
                f"ds_type={self.ds_type}" f"dsize={self.dsize}",
                f"sfrac={self.sfrac}",
                f"nseeds={self.nseeds}",
                f"bs_seed={self.bs_seed}",
                f"nresamples={self.nresamples}",
            ]
        )
        return tag


def generate_data(args: R2W1F2Args) -> np.ndarray:
    ds_type = args.ds_type
    dsize = args.dsize

    if ds_type == "bimodal":
        rng = np.random.default_rng(0)
        datas = []
        p1size = int(0.5 * dsize)
        p2size = dsize - p1size
        dstd = dsize // 100_000
        dloc = dstd * 10
        params_list = [(-dloc, dstd, p1size), (dloc, dstd, p2size)]
        print(f"params_list: {params_list}")
        for params in params_list:
            loc, scale, part_size = params
            part_samples = rng.normal(loc=loc, scale=scale, size=part_size)
            datas.append(part_samples)
        population = np.concatenate(datas)
    elif ds_type == "exponential":
        scale = 0.01
        print(f"Generating exponential data with scale={scale}")
        rng = np.random.default_rng(0)
        population = rng.exponential(scale=100, size=dsize)

    return population


def run(args: R2W1F2Args) -> dict:
    tag = args.get_tag()
    res_path = os.path.join(args.save_dir, f"res-{tag}.pkl")
    if (not args.nocache) and os.path.exists(res_path):
        print(f"Loading from cache: {res_path}")
        res = joblib.load(res_path)
        return res

    sampling_frac = args.sfrac
    nseeds = args.nseeds
    bs_seed = args.bs_seed
    nresamples = args.nresamples

    population = generate_data(args)
    real_median = np.median(population)

    print(f"Population size      : {len(population)}")
    print(f"Population Median    : {real_median}")

    # smethod = 'binomial'
    smethod = "choice"
    rng = np.random.default_rng(0)
    if smethod == "binomial":
        samples = population[
            rng.binomial(1, sampling_frac, len(population)).astype(bool)
        ]
    elif smethod == "choice":
        samples = rng.choice(
            population, size=int(sampling_frac * len(population)), replace=False
        )
    sample_size = len(samples)
    sample_median = np.median(samples)
    print(f"Sampling Fraction : {sampling_frac}")
    print(f"Sampling Size     : {sample_size}")
    print(f"Sample Median     : {sample_median}")

    error_list = []
    for seed in tqdm(range(nseeds), desc="Generating Error List"):
        rng = np.random.default_rng(seed)
        if smethod == "binomial":
            tmp = population[
                rng.binomial(1, sampling_frac, len(population)).astype(bool)
            ]
        elif smethod == "choice":
            tmp = rng.choice(
                population, size=int(sampling_frac * len(population)), replace=False
            )
        error_list.append(np.median(tmp) - real_median)
    error_list = np.array(error_list)

    # test normality of error_list
    _, pval = stats.shapiro(error_list)
    print(f"error_list normality pval={pval}")

    resample_medians = []
    rng = np.random.default_rng(bs_seed)
    for i in tqdm(range(nresamples), desc="Bootstrap Resampling"):
        resample = rng.choice(samples, sample_size, replace=True)
        resample_median = np.median(resample)
        resample_medians.append(resample_median)
    resample_medians = np.array(resample_medians)

    # bias correction
    mean_resample_medians = np.mean(resample_medians)
    bias = mean_resample_medians - sample_median
    bs_median_estimation = sample_median - bias

    print(f"bias                 : {bias}")
    print(f"Resample AVG Median  : {mean_resample_medians}")
    print(f"Bootstrap Median     : {bs_median_estimation}")

    # resample_errors = resample_medians - sample_median
    # resample_errors = resample_medians - mean_resample_medians
    # resample_errors = resample_medians - bs_median_estimation
    resample_errors = resample_medians - real_median

    normal_bs_loc = 0
    normal_bs_std = np.std(resample_medians, ddof=1)
    normal_bs_errors = stats.norm.rvs(
        loc=normal_bs_loc, scale=normal_bs_std, size=10000
    )

    res = {
        "real_median": real_median,
        # "population": population if len(population) <= 100_000 else population[np.random.default_rng(0).choice(len(population), 100_000)],
        "population": population,
        "real_error": sample_median - real_median,
        "error_list": error_list,
        "resample_errors": resample_errors,
        "normal_bs_errors": normal_bs_errors,
    }

    joblib.dump(res, res_path)
    return res


def run_machineryralfsimmedian(args: R2W1F2Args) -> dict:
    seeds_list = np.arange(1000)
    median_qid = 0
    rid_list = [271, 145, 70, 200, 211, 337, 139, 85, 1, 332]
    # rid_list = [rid_list[0]] # pval=0.02
    # rid_list = [rid_list[2]] # pval=0.04
    # rid_list = [rid_list[3]] # pval=0.06
    # rid_list = [rid_list[4]] # pval=0.02
    # rid_list = [rid_list[5]] # pval=0.027
    # rid_list = [rid_list[7]] # pval=0.78
    # rid_list = [rid_list[8]] # pval=0.24
    rid_list = [rid_list[9]] # pval=0.11
    task_name = "machineryralfsimmedian0"
    sim_args = simutils.SimulationArgs().from_dict(
        {"task_name": task_name, "bs_nresamples": args.nresamples}
    )
    ol_args = simutils.get_online_args(sim_args)

    test_set: pd.DataFrame = LoadingHelper.load_dataset(
        ol_args, "test", ol_args.nreqs, offset=ol_args.nreqs_offset
    )
    req_cols = [col for col in test_set.columns if col.startswith("req_")]

    ppl: XIPPipeline = get_ppl(task_name, ol_args, test_set, verbose=False)

    # online_dir = DIRHelper.get_online_dir(ol_args)
    # tag = ppl.settings.__str__()
    # df_path = os.path.join(online_dir, f"final_df_{tag}.csv")
    # assert os.path.exists(df_path), f"File not found: {df_path}"
    # print(f"Loading {df_path}")
    # df = pd.read_csv(df_path)

    fnames = ppl.fextractor.fnames
    noq = ppl.fextractor.num_queries
    qsample_cols = [f"qsamples_{i}" for i in range(noq)]

    requests = test_set[req_cols].to_dict(orient="records")
    res_list = []
    for rid in rid_list:
        request = requests[rid]
        # qsamples = df.loc[rid, qsample_cols].to_numpy()
        # qnparts = (qsamples * 100).astype(int)

        # run exact, make ncores = 0 and loading_mode=1 to be faster
        ppl.fextractor.ncores = 0
        for qry in ppl.fextractor.queries:
            qry.loading_mode = 1
            qry.set_enable_qcache()
            qry.set_enable_dcache()
            qry.profiles = []

        oracle_pred = ppl.serve(request=request, exact=True)
        rrdata: np.ndarray = ppl.fextractor.queries[median_qid]._dcache["cached_rrd"]

        ppl.fextractor.ncores = 1
        for qry in ppl.fextractor.queries:
            qry.loading_mode = 0
            qry.set_enable_qcache()
            qry.set_enable_dcache()
            qry.profiles = []

        xip_pred = ppl.run_apx(request=request, keep_qmc=True)
        qcfgs = ppl.scheduler.get_latest_profile()["qcfgs"]
        qsamples = np.array([qcfg["qsample"] for qcfg in qcfgs])
        qnparts = (qsamples * 100).astype(int)
        print(f'{rid}: {qsamples[median_qid]}, {qnparts[median_qid]}')

        ppl.fextractor.ncores = 1
        for qry in ppl.fextractor.queries:
            qry.loading_mode = 0

        fvecs = []
        for sid, seed in tqdm(
            enumerate(seeds_list), total=len(seeds_list), desc="Running Approximation"
        ):
            qcfgs = ppl.scheduler.get_final_qcfgs(request)
            for qid, qry in enumerate(ppl.fextractor.queries):
                if qid == median_qid:
                    qry.set_enable_qcache()
                    # qry.profiles = qry.profiles[-1:]
                    qry.profiles = []
                    qrng = np.random.default_rng(seed)
                    all_rrd: np.ndarray = rrdata
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

        res_list.append(
            {
                "rid": rid,
                "request": request,
                "oracle_pred": oracle_pred,
                "rrdata": rrdata,
                "xip_pred": xip_pred,
                "fvecs": fvecs,
            }
        )

    rid = 0
    fid = 0

    rrdata = res_list[rid]["rrdata"]
    print(f"rrdata size: {rrdata.shape}")
    print(f"rrdata mean: {np.mean(rrdata)}")
    print(f"rrdata std : {np.std(rrdata)}")
    print(f"rrdata skew: {stats.skew(rrdata)}")
    print(f"rrdata kurt: {stats.kurtosis(rrdata)}")

    real_feature = res_list[rid]["oracle_pred"]["fvec"]["fvals"][fid]
    xip_feature = res_list[rid]["xip_pred"]["fvec"]["fvals"][fid]
    bs_features = res_list[rid]["xip_pred"]["fvec"]["fests"][fid]
    apx_features = np.array([fvec["fvals"][fid] for fvec in fvecs])
    bias = (xip_feature - np.mean(bs_features)) / 2

    print(f"real_feature: {real_feature}")
    print(f"xip_feature : {xip_feature}")
    print(f"bs_features : {np.mean(bs_features)}, {bs_features.shape}")
    print(f"apx_features: {np.mean(apx_features)}")
    print(f"bias        : {bias}")

    apx_errors = np.array(apx_features) - real_feature
    # bs_errors = bs_features - xip_feature
    # bs_errors = bs_features - xip_feature - bias
    # bs_errors = bs_features  - real_feature
    bs_errors = bs_features - np.mean(bs_features)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    ax = axes[0]
    sns.histplot(
        rrdata.flatten(),
        color="blue",
        alpha=0.5,
        bins=100,
        ax=ax,
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Data")

    ax = axes[1]
    sns.kdeplot(
        apx_errors,
        label="Real Error",
        color="blue",
        alpha=0.5,
        ax=ax,
    )
    sns.kdeplot(
        bs_errors,
        label="Bootstrap Error",
        color="red",
        alpha=0.5,
        ax=ax,
    )
    # ref_normal = np.random.default_rng(0).normal(loc = np.mean(apx_errors), scale=np.std(apx_errors), size=10000)
    # sns.kdeplot(
    #     ref_normal,
    #     label="Reference Normal Distribution",
    #     color="red",
    #     alpha=0.5,
    #     ax=ax,
    #     linestyle="--",
    # )

    # ecdf = sm.distributions.ECDF(apx_errors)
    # x = np.linspace(min(apx_errors), max(apx_errors), 1000)
    # y = ecdf(x)
    # ax.step(x, y, label="Real Error", color="blue", alpha=0.5)

    # ecdf = sm.distributions.ECDF(bs_errors)
    # x = np.linspace(min(bs_errors), max(bs_errors), 1000)
    # y = ecdf(x)
    # ax.step(x, y, label="Bootstrap Error", color="red", alpha=0.5)

    # x = np.linspace(min(apx_errors), max(apx_errors), 1000)
    # y = stats.norm.cdf(x, loc=np.mean(apx_errors), scale=np.std(apx_errors))
    # ax.step(x, y, label="Reference Normal Distribution", color="green", alpha=0.5, linestyle="--")

    ax.legend()
    ax.set_xlabel("Error Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Error")

    print(f"normality pval of real error     : {stats.shapiro(apx_errors)[1]}")
    print(f"normality pval of bootstrap error: {stats.shapiro(bs_errors)[1]}")

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(apx_errors, bs_errors)
    print(f"KS statistic: {ks_statistic}, p-value: {p_value}")

    fig_dir = "/home/ckchang/ApproxInfer/revision/cache/figs"
    save_dir = os.path.join(fig_dir, "2.1.2")
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "machineryralfsimmedian.pdf")
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.savefig("./cache/machineryralfsimmedian.png")


def plot_median_error(res: dict):
    print("Plotting Median Error Distribution")
    error_list = res["error_list"]
    resample_errors = res["resample_errors"]

    # plot distribution of error_list
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    ax = axes[0]
    # plot distribution of population
    population = res["population"]
    population = np.random.default_rng(0).choice(population, 100_000)
    sns.histplot(
        population,
        # kde=True,
        label="Data Distribution",
        color="blue",
        alpha=0.5,
        bins=100,
        # stat="density",
        ax=ax,
    )
    ax.axvline(
        res["real_median"],
        color="black",
        linestyle="--",
        label="Median of Data",
    )

    ax.legend()
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Data")

    ax = axes[1]
    # sns.ecdfplot(
    # sns.histplot(
    sns.kdeplot(
        error_list,
        # cumulative=True,
        label="Real Distribution",
        color="blue",
        alpha=0.5,
        ax=ax,
    )

    # sns.ecdfplot(
    # sns.histplot(
    sns.kdeplot(
        resample_errors,
        # cumulative=True,
        label="Bootstrap Distribution",
        color="red",
        alpha=0.5,
        ax=ax,
    )

    ax.legend()
    ax.set_xlabel("Error Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Error")

    # print(f"error_list: {np.unique(error_list, return_counts=True)}")

    print(f"normality pval of real error     : {stats.shapiro(error_list)[1]}")
    print(f"normality pval of bootstrap error: {stats.shapiro(resample_errors)[1]}")

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(error_list, resample_errors)
    print(f"KS statistic: {ks_statistic}, p-value: {p_value}")

    fig_dir = "/home/ckchang/ApproxInfer/revision/cache/figs"
    save_dir = os.path.join(fig_dir, "2.1.2")
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "median_error_dist.pdf")
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.savefig("./cache/median_error_dist.png")


if __name__ == "__main__":
    args = R2W1F2Args().parse_args()
    # res = run(args)
    # plot_median_error(res)
    run_machineryralfsimmedian(args)
