import os
from tap import Tap


class EvalArgs(Tap):
    interpreter: str = "python"
    task_home: str = "final"
    task_name: str = None
    prepare_again: bool = False
    all_nparts: list[int] = [100]
    prep_nparts: int = 100
    all_models: list[str] = ["xgb", "lgbm", "dt", "rf", "lr", "knn", "svm", "mlp"]
    seed: int = 0
    skip_dataset: bool = False

    def process_args(self):
        assert self.task_name is not None


args = EvalArgs().parse_args()

interpreter = args.interpreter
if interpreter != "python":
    interpreter = f"sudo {interpreter}"
TASK_HOME = args.task_home
TASK_NAME = args.task_name
seed = args.seed

for nparts in args.all_nparts:
    command = f"{interpreter} run.py --example {TASK_NAME} --stage ingest --task {TASK_HOME}/{TASK_NAME} --nparts {nparts} --seed {seed}"
    os.system(command=command)

nparts = args.prep_nparts
command = f"{interpreter} run.py --example {TASK_NAME} --stage prepare --task {TASK_HOME}/{TASK_NAME} --nparts {nparts} --seed {seed}"
if args.prepare_again:
    if args.skip_dataset:
        command += " --skip_dataset"
    os.system(command=command)

for model in args.all_models:
    command = f"{interpreter} run.py --example {TASK_NAME} --stage train --task {TASK_HOME}/{TASK_NAME} --model {model} --nparts {nparts} --seed {seed}"
    os.system(command=command)
