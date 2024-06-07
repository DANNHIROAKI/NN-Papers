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

# disable warning
warnings.filterwarnings("ignore")

db_client = DBHelper.get_db_client()
database = "xip_0"
table = "trips_100"
cls_score = "acc"
reg_score = "r2"


task_name = "machineryralf"
nops: int = 8
naggs: int = 8
agg_ids: List[int] = list(range(8))
is_aggop: List[bool] = [True] * 8

settings = {
    "task": "final/machineryralf",
    "model": "mlp",
    "seed": 0,
    "pest": "biathlon",
    "pest_nsamples": 128,
    "scheduler_init": 5,
    "scheduler_batch": 1 * naggs,
    "max_error": 0.0,
    "min_conf": 0.95,
}

args: OnlineArgs = OnlineArgs().from_dict(settings)
model = LoadingHelper.load_model(args)
model_home = os.path.dirname(DIRHelper.get_model_path(args))
fimp = pd.read_csv(os.path.join(model_home, "mlp_feature_importance.csv"))
# print(fimp)

test_set = LoadingHelper.load_dataset(
    args, "test", args.nreqs, offset=args.nreqs_offset
)
fnames = [col for col in test_set.columns if col.startswith("f_")]
fops = [fname.split("_")[-1] for fname in fnames]
opnames = sorted(list(set(fops)))
# print(f"nops: {nops}")
# print(f"fnames: {fnames}")
# print(f"opnames: {opnames}")

features = test_set[fnames].values
targets = test_set["label"].values

default_dsize = 100_000 + 1
default_dsizes = [default_dsize] * nops
default_dists = ["reviewer"] * nops
default_synv = -100
default_dist_args = [(0.5 + 1e-5, default_synv) for i in range(nops)]
default_operators = [None] * nops
default_opid = 0
default_operators[default_opid] = ["median"]
default_dbseed = 0
default_rid = 0


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
        p, synv = kwargs.get("arg")
        x_num = int(dsize * p)
        data = np.array([x] * x_num + [synv] * (dsize - x_num))
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
    dsizes: List[int] = default_dsizes,
    ddists: List[int] = default_dists,
    dist_args: list = default_dist_args,
    operators: List[List[str] | None] = default_operators,
    dbseed: int = default_dbseed,
    rid: int = default_rid,
    keep_latency: bool = False,
) -> dict:
    request = test_set.iloc[rid].to_dict()
    x = features[rid]
    y_true = targets[rid]
    y_exact = ppl.model.predict(x.reshape(1, -1))[0]

    dataset = {
        qry.qname: generate_synthetic_data(
            x[qid],
            operators[qid][0],
            dsizes[qid],
            dbseed,
            ddists[qid],
            arg=dist_args[qid],
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


class DebugArgs(Tap):
    debug: bool = False
    logfile: str = "cache/debug.log"
    pest: str = "biathlon"
    pest_nsamples: int = 128
    bs_nresamples: int = 128
    min_conf: float = 0.95
    init: int = 5
    batch: int = 1 * naggs
    rid: int = 0


debug_args = DebugArgs().parse_args()
tmp_args: OnlineArgs = OnlineArgs().from_dict(settings)
tmp_args.scheduler_init = debug_args.init
tmp_args.scheduler_batch = debug_args.batch
tmp_args.pest = debug_args.pest
tmp_args.pest_nsamples = debug_args.pest_nsamples
tmp_args.bs_type = "descrete"
tmp_args.bs_nresamples = debug_args.bs_nresamples
tmp_args.bs_feature_correction = True
tmp_args.bs_bias_correction = True
tmp_args.min_conf = debug_args.min_conf
ppl: XIPPipeline = get_ppl(task_name, tmp_args, test_set, verbose=False)

if debug_args.debug:
    set_logger(logging.DEBUG, debug_args.logfile)
res = run(ppl, keep_latency=True, rid=debug_args.rid)
set_logger(logging.INFO)
print(
    len(res["history"]),
    res["y_exact"] == res["xip_pred"]["pred_value"],
    res["latency"],
    [qcfg["qsample"] for qcfg in res["history"][-1]["qcfgs"]],
)

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
        "r" if is_same_float(pred_value_trace[i], res["y_exact"]) else "b"
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
        "r" if is_same_float(pred_value_trace[i], res["y_exact"]) else "b"
        for i in range(len(pred_value_trace))
    ],
)
axes[1].set_title("Prediction Value")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Value")

axes[2].plot(fval_trace[:, default_opid], label=f"f_{default_opid}")
axes[2].scatter(x, fval_trace[:, default_opid])
axes[2].set_title("Feature Value")
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Feature Value")
axes[2].legend()
axes[2].grid()

for i in range(nops):
    if i == default_opid:
        continue
    axes[3].plot(fval_trace[:, i], label=f"f_{i}")
    axes[3].scatter(x, fval_trace[:, i])
axes[3].set_title("Feature Value")
axes[3].set_xlabel("Iteration")
axes[3].set_ylabel("Feature Value")
axes[3].legend()
axes[3].grid()

axes[4].plot(qsamples_trace[:, default_opid], label=f"f_{default_opid}")
axes[4].scatter(x, qsamples_trace[:, default_opid])
axes[4].set_title("Query Sample Frac")
axes[4].set_xlabel("Iteration")
axes[4].set_ylabel("Query Sample Frac")
# axes[4].set_yscale("log")
axes[4].legend()
axes[4].grid()

for i in range(nops):
    if i == default_opid:
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
plt.savefig("./cache/trace.png")
