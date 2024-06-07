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


task_meta = {
    "machineryralf": {
        "nops": 8,
        "naggs": 8,
        "agg_ids": list(range(8)),
        "is_aggop": [True] * 8,
        "model": "mlp",
        "max_error": 0.0,
        "nreqs": 338,
    },
    "tickralfv2": {
        "nops": 7,
        "naggs": 1,
        "agg_ids": [6],
        "is_aggop": [False] * 6 + [True],
        "model": "lr",
        "max_error": 0.04,
        "nreqs": 4740,
    },
    "turbofan": {
        "nops": 9,
        "naggs": 9,
        "agg_ids": list(range(9)),
        "is_aggop": [True] * 9,
        "model": "rf",
        "max_error": 4.88,
        "nreqs": 769,
    },
    "tripsralfv2": {
        "nops": 3,
        "naggs": 2,
        "agg_ids": [1, 2],
        "is_aggop": [False, True, True],
        "model": "lgbm",
        "max_error": 1.5,
        "nreqs": 22016,
    },
    "tripsralfv3": {
        "nops": 3,
        "naggs": 2,
        "agg_ids": [1, 2],
        "is_aggop": [False, True, True],
        "model": "lgbm",
        "max_error": 1.4,
        "nreqs": 1960,
    },
    "tdfraudralf2d": {
        "nops": 4,
        "naggs": 3,
        "agg_ids": [1, 2, 3],
        "is_aggop": [False, True, True, True],
        "model": "xgb",
        "max_error": 0.0,
        "nreqs": 8603,
    },
    "studentqnov2subset": {
        "nops": 13,
        "naggs": 13,
        "agg_ids": list(range(13)),
        "is_aggop": [True] * 13,
        "model": "rf",
        "max_error": 0.0,
        "nreqs": 471,
    },
    "batteryv2": {
        "nops": 6,
        "naggs": 5,
        "agg_ids": list(range(5)),
        "is_aggop": [True] * 5 + [False],
        "model": "lgbm",
        "max_error": 189.0,
        "nreqs": 564,
    },
}

median_tasks = {
    **{
        f"machineryralfsimmedian"
        + "".join([f"{j}" for j in range(i)]): {**task_meta["machineryralf"]}
        for i in range(9)
    },
    **{
        f"machineryralfe2emedian"
        + "".join([f"{j}" for j in range(i)]): {**task_meta["machineryralf"]}
        for i in range(9)
    },
    **{
        f"machineryralfdirectmedian"
        + "".join([f"{j}" for j in range(i)]): {**task_meta["machineryralf"]}
        for i in range(9)
    },
}
for name, meta in task_meta.items():
    median_tasks[f"{name}median"] = {**meta}
    median_tasks[f"{name}simmedian"] = {**meta}
    median_tasks[f"{name}v2median"] = {**meta}
task_meta.update(median_tasks)

task_meta["batteryv2median"]["max_error"] = 186.7
task_meta["batteryv2simmedian"]["max_error"] = 186.7
task_meta["tripsralfv2median"]["max_error"] = 1.4
task_meta["tripsralfv2simmedian"]["max_error"] = 1.4


class SimulationArgs(Tap):
    task_home: str = "final"
    task_name: str = "machineryralf"

    bs_type: str = "descrete"
    bs_nresamples: int = 100

    pest: str = "biathlon"
    pest_nsamples: int = 128
    pest_seed: int = 0
    min_conf: float = 0.95

    alpha: int = 5
    beta: int = 1

    dsize: int = 100_000 + 1
    synv: float = -100.0
    dist: str = "reviewer"
    dist_param: float = 1e-5

    op_type: str = "median"
    selected_qid: int = 0
    seed: int = 0
    syndb_seed: int = 0
    rid: int = 0

    scaling_factor: float = 1.0

    keep_latency: bool = False

    debug: bool = False
    save_dir: str = "/home/ckchang/ApproxInfer/cache"
    logfile: str = "debug.log"

    nocache: bool = False

    def process_args(self) -> None:
        if self.task_name == "tickralfv2":
            assert self.selected_qid == 6, "Only support selected_qid=6 for tickralfv2"

    def get_tag(self):
        tag = "_".join(
            [
                self.task_name,
                f"{self.rid}",
                f"{self.selected_qid}",
                f"{self.op_type}",
                f"{self.dist}",
                f"{self.synv}",
                f"{self.dist_param}",
                f"{self.dsize}",
                f"{self.seed}",
                f"{self.syndb_seed}",
                f"{self.bs_type}",
                f"{self.bs_nresamples}",
                f"{self.pest}",
                f"{self.pest_nsamples}",
                f"{self.pest_seed}",
                f"{self.min_conf}",
                f"{self.alpha}",
                f"{self.beta}",
                f"{self.scaling_factor}",
            ]
        )
        return tag


def generate_synthetic_data(
    x: float, operator: str, dsize: int, seed: int, ddist: str, **kwargs
) -> np.ndarray:
    # generate {dsize} synthetic data that
    # following {ddsit} distribution with {dist_args}
    # make sure that x == {operator}(data)
    rng = np.random.default_rng(0)
    if ddist == "norm":
        std_value = kwargs.get("arg")
        data = rng.normal(x, std_value, dsize)
    elif ddist == "uni":
        half_range = kwargs.get("arg")
        data = rng.uniform(x - half_range, x + half_range, dsize)
    elif ddist == "zipf":
        alpha = kwargs.get("arg")
        data = rng.zipf(alpha, dsize).astype(float)
        data = np.minimum(data, dsize)
    elif ddist == "reviewer":
        p = kwargs.get("arg")
        synv = kwargs.get("synv")
        x_num = int(dsize * (0.5 + p))
        data = np.array([x] * x_num + [synv] * (dsize - x_num))
    elif ddist == "bimodal":
        p = kwargs.get("arg")
        rng = np.random.default_rng(0)
        datas = []
        p1size = int(0.5 * dsize)
        p2size = dsize - p1size
        dstd = dsize // 100_000
        dloc = dstd * 10
        params_list = [(-dloc, dstd, p1size), (dloc, dstd, p2size)]
        # print(f"params_list: {params_list}")
        for params in params_list:
            loc, scale, part_size = params
            part_samples = rng.normal(loc=loc, scale=scale, size=part_size)
            datas.append(part_samples)
        data = np.concatenate(datas)
    else:
        raise ValueError(f"Invalid distribution: {ddist}")

    if operator == "mean":
        if not is_same_float(x, np.mean(data)):
            data = data + (x - np.mean(data))
        assert is_same_float(x, np.mean(data)), f"{x} != {np.mean(data)}"
    elif operator == "median":
        if not is_same_float(x, np.median(data)):
            data = data + (x - np.median(data))
        assert is_same_float(x, np.median(data)), f"{x} != {np.median(data)}"
    elif operator == "max":
        if not is_same_float(x, np.max(data)):
            data = data + (x - np.max(data))
        assert is_same_float(x, np.max(data)), f"{x} != {np.max(data)}"
    elif operator == "min":
        if not is_same_float(x, np.min(data)):
            data = data + (x - np.min(data))
        assert is_same_float(x, np.min(data)), f"{x} != {np.min(data)}"
    else:
        raise ValueError(f"Invalid aggregation type: {operator}")

    # shuffule data
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    return data


def run(
    ppl: XIPPipeline,
    request: dict,
    dsizes: List[int],
    ddists: List[int],
    dist_args: List[float],
    operators: List[List[str] | None],
    syndb_seed: int,
    synv: float,
    keep_latency: bool = False,
    run_exact: bool = False,
) -> dict:
    fnames = [f"f_{fname}" for fname in ppl.fextractor.fnames]
    y_true = request["label"]

    if run_exact:
        xip_pred = ppl.serve(request, exact=True)
        x = xip_pred["fvec"]["fvals"]
        y_exact = xip_pred["pred_value"]
        for qid, qry in enumerate(ppl.fextractor.queries):
            qry.set_enable_qcache()
            qry.set_enable_dcache()
            qry.profiles = []
    else:
        x = np.array([request[fname] for fname in fnames])
        y_exact = ppl.model.predict(x.reshape(1, -1))[0]

    dataset = {
        qry.qname: generate_synthetic_data(
            x[qid],
            operators[qid][0],
            dsizes[qid],
            syndb_seed,
            ddists[qid],
            arg=dist_args[qid],
            synv=synv,
        ).reshape(-1, 1)
        for qid, qry in enumerate(ppl.fextractor.queries)
        if operators[qid] is not None
    }
    request["syn_data"] = dataset
    if keep_latency:
        request["keep_latency"] = True

    for qid, qry in enumerate(ppl.fextractor.queries):
        for i in range(len(qry.qops)):
            if operators[qid] is not None:
                qry.qops[i]["dops"] = operators[qid]

    xip_pred = ppl.serve(request, exact=False)
    latency = time.time() - ppl.start_time

    for qid, qry in enumerate(ppl.fextractor.queries):
        qry.set_enable_qcache()
        qry.set_enable_dcache()
        qry.profiles = []

    return {
        "xip_pred": xip_pred,
        "feature": x,
        "y_true": y_true,
        "y_exact": y_exact,
        "history": ppl.scheduler.history,
        "latency": latency,
        "request": request,
    }


def set_logger(level: int = logging.INFO, log_file: str = None):
    # set logging level for all registered loggers
    import logging

    for key in logging.Logger.manager.loggerDict:
        if level == logging.DEBUG:
            if "XIP" in key or "apxinf" in key:
                print(f"Set {key} to {level}")
                logging.getLogger(key).setLevel(level)
        else:
            logging.getLogger(key).setLevel(level)
        # set logging file
        if log_file is not None:
            print(f"Log to {log_file}")
            fh = logging.FileHandler(filename=log_file)
            logging.getLogger(key).addHandler(fh)
            logging.getLogger(key).propagate = False


def get_online_args(args: SimulationArgs) -> OnlineArgs:
    meta = task_meta[args.task_name]
    settings = {
        "task": os.path.join(args.task_home, args.task_name),
        "model": meta["model"],
        "seed": args.seed,
        "bs_type": args.bs_type,
        "bs_nresamples": args.bs_nresamples,
        "pest": args.pest,
        "pest_nsamples": args.pest_nsamples,
        "pest_seed": args.pest_seed,
        "scheduler_init": args.alpha,
        "scheduler_batch": meta["naggs"] * args.beta,
        "max_error": meta["max_error"],
        "min_conf": args.min_conf,
    }
    ol_args = OnlineArgs().from_dict(settings)
    return ol_args


def pred_check(task_name: str, y_pred, y_oracle) -> bool:
    max_error = task_meta[task_name]["max_error"]
    return np.abs(y_pred - y_oracle) < max_error + 1e-9


def run_default(
    sim_args: SimulationArgs, verbose: bool = False, run_exact: bool = False
) -> dict:
    tag = sim_args.get_tag()
    os.makedirs(os.path.join(sim_args.save_dir, "results"), exist_ok=True)
    res_path = os.path.join(sim_args.save_dir, "results", f"res_{tag}.pkl")
    if not sim_args.nocache:
        if os.path.exists(res_path):
            if verbose:
                print(f"Load result from {res_path}")
            res = joblib.load(res_path)
            return res

    # print(f'sim_args: {sim_args}')
    args: OnlineArgs = get_online_args(sim_args)
    # print(f'args: {args}')

    test_set: pd.DataFrame = LoadingHelper.load_dataset(
        args, "test", args.nreqs, offset=args.nreqs_offset
    )
    # fnames = [col for col in test_set.columns if col.startswith("f_")]
    # features = test_set[fnames].values
    # targets = test_set["label"].values

    ppl: XIPPipeline = get_ppl(sim_args.task_name, args, test_set, verbose=False)
    if sim_args.debug:
        set_logger(logging.DEBUG, os.path.join(sim_args.save_dir, sim_args.logfile))

    request = test_set.iloc[sim_args.rid].to_dict()
    dsizes = [sim_args.dsize] * len(ppl.fextractor.queries)
    ddists = [sim_args.dist] * len(ppl.fextractor.queries)
    dist_args = [sim_args.dist_param] * len(ppl.fextractor.queries)
    operators = [
        [sim_args.op_type] if i == sim_args.selected_qid else None
        for i in range(len(ppl.fextractor.queries))
    ]

    res = run(
        ppl=ppl,
        request=request,
        dsizes=dsizes,
        ddists=ddists,
        dist_args=dist_args,
        operators=operators,
        syndb_seed=sim_args.syndb_seed,
        synv=sim_args.synv,
        keep_latency=sim_args.keep_latency,
        run_exact=run_exact,
    )
    if verbose:
        print(
            len(res["history"]),
            pred_check(
                sim_args.task_name, res["xip_pred"]["pred_value"], res["y_exact"]
            ),
            res["latency"],
            [qcfg["qsample"] for qcfg in res["history"][-1]["qcfgs"]],
        )

    joblib.dump(res, res_path)
    if verbose:
        print(f"Saved result to {res_path}")

    return res


def run_qmc(
    sim_args: SimulationArgs,
    verbose: bool = False,
) -> dict:
    args: OnlineArgs = get_online_args(sim_args)
    task_name = sim_args.task_name
    meta = task_meta[task_name]
    rid = sim_args.rid
    scaling_factor = sim_args.scaling_factor

    os.makedirs(os.path.join(sim_args.save_dir, "qmc"), exist_ok=True)
    tag = sim_args.get_tag()
    res_path = os.path.join(sim_args.save_dir, "qmc", f"res_{tag}.pkl")
    if not sim_args.nocache:
        if os.path.exists(res_path):
            if verbose:
                print(f"Load result from {res_path}")
            res = joblib.load(res_path)
            return res

    test_set: pd.DataFrame = LoadingHelper.load_dataset(
        args, "test", args.nreqs, offset=args.nreqs_offset
    )

    ppl = get_ppl(task_name, args, test_set, verbose=False)
    assert isinstance(ppl.pred_estimator, BiathlonPredictionEstimator)

    fnames = ppl.fextractor.fnames
    features = test_set[[f"f_{col}" for col in fnames]].values
    targets = test_set["label"].values
    fstds = np.std(features, axis=0)

    rid = sim_args.rid
    request = test_set.iloc[rid].to_dict()
    x = features[rid]
    y_true = targets[rid]
    y_exact = ppl.model.predict(x.reshape(1, -1))[0]
    if verbose:
        print(f"(rid={rid}) y_true: {y_true}, y_exact: {y_exact}")

    fests = np.zeros_like(fstds)
    fdists = ["fixed"] * len(ppl.fextractor.fnames)
    for qid in meta["agg_ids"]:
        for fname in ppl.fextractor.queries[qid].fnames:
            fid = fnames.index(fname)
            fests[fid] = fstds[fid] * scaling_factor
            fdists[fid] = "normal"
    fvecs = XIPFeatureVec(
        fnames=fnames,
        fvals=x,
        fests=fests,
        fdists=fdists,
    )

    ppl.pred_estimator.n_samples = sim_args.pest_nsamples
    ppl.pred_estimator.seed = sim_args.pest_seed
    qmc_preds = ppl.pred_estimator.get_qmc_preds(ppl.model, fvecs)
    res = {
        "qmc_preds": qmc_preds,
        "pred_mean": np.mean(qmc_preds),
        "pred_std": np.std(qmc_preds),
        "nsamples": sim_args.pest_nsamples,
        "seed": sim_args.pest_seed,
        "rid": rid,
        "fvals": x,
        "fstds": fstds,
        "y_exact": y_exact,
        "y_true": y_true,
    }

    joblib.dump(res, res_path)
    if verbose:
        print(f"Saved result to {res_path}")

    return res


if __name__ == "__main__":
    sim_args = SimulationArgs().parse_args()
    run_default(sim_args)
