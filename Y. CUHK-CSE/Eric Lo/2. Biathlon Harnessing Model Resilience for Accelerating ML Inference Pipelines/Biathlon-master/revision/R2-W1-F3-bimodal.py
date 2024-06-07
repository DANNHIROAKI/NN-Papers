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


class R2W1F3Args(simutils.SimulationArgs):
    save_dir: str = "/home/ckchang/ApproxInfer/revision/cache"
    max_nreqs: int = None


if __name__ == "__main__":
    task_meta = simutils.task_meta
    sim_args = R2W1F3Args().parse_args()

    nreqs = simutils.task_meta[sim_args.task_name]["nreqs"]
    if sim_args.max_nreqs is not None:
        nreqs = min(nreqs, sim_args.max_nreqs)
    if sim_args.task_name == "tickralfv2":
        nreqs = min(nreqs, 100)
    rid_list = np.arange(nreqs)

    task_name = sim_args.task_name
    tag = f"{sim_args.get_tag()}_rid0-{nreqs-1}"

    res_list_dir = os.path.join(sim_args.save_dir, "results", "2.1.3.sim")
    os.makedirs(res_list_dir, exist_ok=True)
    res_list_path = os.path.join(res_list_dir, f"res_list_{tag}.pkl")
    if os.path.exists(res_list_path):
        print(f"Loading res_list from {res_list_path}")
        res_list = joblib.load(res_list_path)
    else:
        res_list = []
        sim_args = R2W1F3Args().parse_args()
        sim_args.dist = "bimodal"
        sim_args.dsize = 10_000_000 + 1
        sim_args.dist_param = 0.5
        for rid in tqdm(rid_list, desc="rid", leave=False):
            sim_args.rid = rid
            res = simutils.run_default(sim_args)
            res_list.append(res)
        joblib.dump(res_list, res_list_path)

    sim_args = R2W1F3Args().parse_args()
    task_name = sim_args.task_name
    nops = simutils.task_meta[task_name]["nops"]
    agg_ids = simutils.task_meta[task_name]["agg_ids"]
    selected_qid = sim_args.selected_qid
    qsamples_list = np.array(
        [[qcfg["qsample"] for qcfg in res["history"][-1]["qcfgs"]] for res in res_list]
    )
    non_agg_ids = np.setdiff1d(np.arange(nops), agg_ids)
    qsamples_list[:, non_agg_ids] = 0.0
    avg_samples_list = np.array(
        [np.mean(qsamples[agg_ids]) for qsamples in qsamples_list]
    )
    perror_list = np.array(
        [abs(res["xip_pred"]["pred_value"] - res["y_exact"]) for res in res_list]
    )
    ferrors_list = np.array(
        [np.abs(res["xip_pred"]["fvec"]["fvals"] - res["feature"]) for res in res_list]
    )
    latency_list = np.array([res["latency"] for res in res_list])
    last_hist_list = [res["history"][-1] for res in res_list]

    print(f"task_name: {task_name}")
    print(f"tag: {tag}")
    print(f"nreqs: {nreqs}")
    print(f"selected_qid: {selected_qid}")
    print(f"agg_ids: {agg_ids}")
    print(f"non_agg_ids: {non_agg_ids}")
    print(f"qsamples_list.shape: {qsamples_list.shape}, {qsamples_list}")
    print(f"avg_samples_list.shape: {avg_samples_list.shape}, {avg_samples_list}")
    print(f"perror_list.shape: {perror_list.shape}, {perror_list}")
    print(f"ferrors_list.shape: {ferrors_list.shape}, {ferrors_list}")
    print(f"latency_list.shape: {latency_list.shape}, {latency_list}")
    print(f"last_hist_list: {len(last_hist_list)}")
    print(f"last_hist_list: {last_hist_list}")
