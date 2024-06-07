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

if __name__ == "__main__":
    # sim_args = simutils.SimulationArgs().parse_args()
    # ol_args = simutils.get_online_args(sim_args)
    ol_args = OnlineArgs().parse_args()
    online_dir = DIRHelper.get_online_dir(ol_args)
    tag = DIRHelper.get_eval_tag(ol_args)
    evals_path = os.path.join(online_dir, f"evals_{tag}.json")
    assert os.path.exists(evals_path), f"File not found: {evals_path}"
    print(f"Loading {evals_path}")
    with open(evals_path, "r") as f:
        evals = json.load(f)
    avg_sample_query = np.array(evals['avg_sample_query'])
    print(avg_sample_query)

    task_name = ol_args.task.split("/")[-1]
    meta = simutils.task_meta[task_name]
    agg_ids = meta["agg_ids"]
    avg_samples = np.mean(avg_sample_query[agg_ids])
    print(f"Average samples: {avg_samples}")

    # sudo /home/ckchang/anaconda3/envs/apx/bin/python get_avg_samples.py --task servers/2024041308/ssd16/tripsralfv2 --model lgbm --max_error 1.5 --scheduler_batch 2
    # sudo /home/ckchang/anaconda3/envs/apx/bin/python get_avg_samples.py --task final/tripsralfv2 --model lgbm --max_error 1.5 --scheduler_batch 2
    # sudo /home/ckchang/anaconda3/envs/apx/bin/python get_avg_samples.py --task final/tickralfv2 --model lr --max_error 0.04 --scheduler_batch 1 --nreqs 4740  # ssd8
    # sudo /home/ckchang/anaconda3/envs/apx/bin/python get_avg_samples.py --task final/batteryv2 --model lgbm --max_error 189.0 --scheduler_batch 5 --nreqs 564  # ssd18
    # sudo /home/ckchang/anaconda3/envs/apx/bin/python get_avg_samples.py --task final/turbofan --model rf --max_error 4.88 --scheduler_batch 9 --nreqs 769  # ssd6
    # sudo /home/ckchang/anaconda3/envs/apx/bin/python get_avg_samples.py --task final/tdfraudralf2d --model xgb --scheduler_batch 3 --nreqs 8603  # ssd6
    # sudo /home/ckchang/anaconda3/envs/apx/bin/python get_avg_samples.py --task final/machineryralf --model mlp --scheduler_batch 8 --nreqs 338  # ssd17
    # sudo /home/ckchang/anaconda3/envs/apx/bin/python get_avg_samples.py --task final/studentqnov2subset --model rf --scheduler_batch 13 --nreqs 471  # ssd16
