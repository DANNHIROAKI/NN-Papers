import os
import json
import pandas as pd
from tap import Tap

from apxinfer.core.config import EXP_HOME

from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS


class EvalArgs(Tap):
    interpreter: str = "python"
    task_home: str = "final"
    task_name: str = None
    nparts: int = 100
    ncfgs: int = None
    ncores: int = 1
    loading_mode: int = 0
    seed: int = 0
    agg_qids: list[int] = None
    model: str = None

    pest: str = "biathlon"
    pest_constraint: str = "error"
    pest_nsamples: int = 1000
    pest_seed: int = 0

    qinf: str = "biathlon"

    policy: str = "optimizer"
    scheduler_init: int = 1
    scheduler_batch: int = 1
    nocache: bool = False
    max_error: float = 1.0
    min_confs: list[float] = [
        0.999,
        0.995,
        0.99,
        0.98,
        0.95,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.0,
    ]

    ralf_budget: float = 1.0

    run_shared: bool = False
    run_offline: bool = False
    run_baseline: bool = False

    def process_args(self):
        assert self.task_name is not None
        assert self.model is not None
        assert self.agg_qids is not None


args = EvalArgs().parse_args()
args.process_args()

EVALS_HOME = "./evals"
interpreter = args.interpreter
if interpreter != "python":
    interpreter = f"sudo {interpreter}"
TASK_HOME = args.task_home
TASK_NAME = args.task_name
model = args.model
nparts = args.nparts
if args.ncfgs is None:
    ncfgs = nparts
else:
    ncfgs = args.ncfgs
ncores = args.ncores
loading_mode = args.loading_mode
seed = args.seed
pest_seed = args.pest_seed

pest = args.pest
pest_constraint = args.pest_constraint
pest_nsamples = args.pest_nsamples

qinf = args.qinf

policy = args.policy
scheduler_init = args.scheduler_init
scheduler_batch = args.scheduler_batch
max_error = args.max_error
min_confs = args.min_confs

offline_nreqs = 100
if nparts >= 20:
    offline_nreqs = 50

agg_qids = args.agg_qids
ralf_budget = args.ralf_budget


def extract_result(all_info: dict, min_conf, base_time=None):
    avg_smpl_rate = 0.0
    for qid in agg_qids:
        avg_smpl_rate += all_info["avg_sample_query"][qid]
    avg_smpl_rate /= len(agg_qids)

    result = {
        **args.as_dict(),
        "min_conf": min_conf,
        "sampling_rate": avg_smpl_rate,
        "avg_latency": all_info["avg_ppl_time"],
        "speedup": base_time / all_info["avg_ppl_time"]
        if base_time is not None
        else 1.0,
        "BD:AFC": all_info["avg_query_time"],
        "BD:AMI": all_info["avg_pred_time"],
        "BD:Sobol": all_info["avg_scheduler_time"],
        "avg_nrounds": all_info["avg_nrounds"],
        "avg_sample_query": all_info["avg_sample_query"],
        "avg_qtime_query": all_info["avg_qtime_query"],
        "meet_rate": all_info["meet_rate"],
        "avg_real_error": all_info["avg_real_error"],
    }
    dropped_keys = [
        "run_shared",
        "run_offline",
        "run_baseline",
        "nocache",
        "interpreter",
        "min_confs",
    ]
    for key in dropped_keys:
        result.pop(key, None)
    for key in all_info["evals_to_ext"]:
        if key in ["time", "size"]:
            continue
        result[f"similarity-{key}"] = all_info["evals_to_ext"][key]
    for key in all_info["evals_to_gt"]:
        if key in ["time", "size"]:
            continue
        result[f"accuracy-{key}"] = all_info["evals_to_gt"][key]
    if args.task_name in ALL_REG_TASKS:
        result["similarity"] = result["similarity-r2"]
        result["accuracy"] = result["accuracy-r2"]
    else:
        assert args.task_name in ALL_CLS_TASKS, f"unknown task: {args.task_name}"
        result["similarity"] = result["similarity-f1"]
        result["accuracy"] = result["accuracy-f1"]
    return result


shared_cmd = f"{interpreter} run.py --example {TASK_NAME} --task {TASK_HOME}/{TASK_NAME} --model {model} --nparts {nparts} --offline_nreqs {offline_nreqs} --ncfgs {ncfgs} --ncores {ncores} --loading_mode {loading_mode} --seed {seed}"
shared_cmd = f"{shared_cmd} --ralf_budget {ralf_budget}"
offline_cmd = f"{shared_cmd} --stage offline"
online_cmd = f"{shared_cmd} --stage online"

shared_prefix = f"{EXP_HOME}/{TASK_HOME}/{TASK_NAME}/seed-{seed}"
qcm_path = f"{shared_prefix}/offline/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}/ncfgs-{ncfgs}/nreqs-{offline_nreqs}/model/xip_qcm.pkl"
online_prefix = f"{shared_prefix}/online/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}"
exact_path = f"{online_prefix}/exact/evals_exact.json"

if args.run_offline or args.run_shared:
    # offline
    command = f"{offline_cmd} --clear_cache"
    print(command)
    os.system(command=command)

if args.run_baseline or args.run_shared:
    # exact
    command = f"{online_cmd} --exact"
    print(command)
    os.system(command=command)

if (not args.run_shared) and (not args.run_offline) and (not args.run_baseline):
    evals = []
    evals.append(extract_result(json.load(open(exact_path)), 1.0))
    print(f"last eval: {evals[-1]}")

    for min_conf in min_confs:
        pest_opts = f"--pest {pest} --pest_constraint {pest_constraint} --pest_seed {pest_seed} --pest_nsamples {pest_nsamples}"
        qinf_opts = f"--qinf {qinf}"
        scheduler_opts = f"--scheduler {policy} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch}"
        acc_opts = f"--max_error {max_error} --min_conf {min_conf}"
        command = f"{online_cmd} {pest_opts} {qinf_opts} {scheduler_opts} {acc_opts}"
        eval_path = f"{online_prefix}/ncfgs-{ncfgs}/pest-{pest_constraint}-{pest}-{pest_nsamples}-{pest_seed}/qinf-{qinf}/scheduler-{policy}-{scheduler_init}-{scheduler_batch}/evals_conf-0.05-{max_error}-{min_conf}-60.0-2048.0-1000.json"
        print(f"path: {eval_path}")
        if args.nocache or (not os.path.exists(eval_path)):
            os.system(command=command)
        evals.append(
            extract_result(
                json.load(open(eval_path)), min_conf, evals[0]["avg_latency"]
            )
        )
        print(f"last eval: {evals[-1]}")

    evals_dir = os.path.join(EVALS_HOME, f"seed-{seed}", f"mode-{loading_mode}")
    if not os.path.exists(evals_dir):
        os.makedirs(evals_dir, exist_ok=True)

    # conver evals to pd.DataFrame and save as csv
    csv_path = f"{evals_dir}/{TASK_NAME}_{qinf}-{policy}-{scheduler_init}-{scheduler_batch}-{pest_nsamples}_{model}_{nparts}_{ncfgs}_{ncores}_{max_error}.csv"
    evals = pd.DataFrame(evals)
    evals.to_csv(csv_path, index=False)
