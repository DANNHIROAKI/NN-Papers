from tap import Tap
import os
from typing import List, Literal

from apxinfer.core.config import EXP_HOME

StudentQNo = [f"studentqno{i}" for i in range(1, 19)]
StudentQNo18VaryNF = [f"studentqno18nf{i}" for i in range(1, 13)]
MachineryVaryNF = [f"machinerynf{i}" for i in range(1, 8)]
MachineryVaryXNF = [f"machineryxf{i}" for i in range(1, 9)]
MachineryMultiVaryNF = [f"machinerymultif{i}" for i in range(1, 8)]
MachineryMultiVaryXNF = [f"machinerymultixf{i}" for i in range(1, 9)]
MachineryRalfVaryNF = [f"machineryralfnf{i}" for i in range(1, 8)]
TickVaryNMonths = [f"tickvaryNM{i}" for i in range(1, 30)]
TripsFeastVaryWindow = [f"tripsfeastw{i}" for i in range(1, 1000)]


class ExpArgs(Tap):
    interpreter = "/home/ckchang/anaconda3/envs/apx/bin/python"
    version: str = "latest"
    task_home: str = "final"
    exp: str = None
    phase: str = "biathlon"

    seed: int = 0
    model: str = None  # see each exp
    ncores: int = 1  # 1, 0
    loading_mode: int = 0  # 0, 1, 2, 5, 10
    nparts: int = 100
    skip_dataset: bool = False

    ncfgs: int = 100
    offline_nreqs: int = 50

    bs_nthreads: int = 1  # nthreads for bootstrapping
    bs_type: Literal["descrete", "fstd", "allbs"] = "fstd"
    bs_nresamples: int = 100
    bs_feature_correction: bool = True
    bs_bias_correction: bool = True
    bs_for_var_std: bool = True

    pest: str = "biathlon"
    pest_seed: int = 0
    pest_nsamples: int = 128
    qinf: str = "biathlon"
    scheduler: str = "optimizer"

    ralf_budget: float = 1.0
    ralf_budgets: List[float] = None

    default_alpha: int = 5
    default_beta: int = 1
    default_error: float = 0.0
    default_conf: float = 0.95

    default_only: bool = False
    nocache: bool = False

    try_execution: bool = False

    test_alpha: int = 5
    test_beta: int = 1
    test_error: float = 0.0
    test_conf: float = 0.95

    def process_args(self):
        assert self.exp is not None
        assert self.model is not None

        if self.interpreter != "python":
            self.interpreter = f"sudo {self.interpreter}"

        if self.version == "submission":
            self.pest = "MC"
            self.pest_nsamples = 1000
            self.pest_seed = self.seed
            self.qinf = "sobol"
        elif self.version == "revision":
            self.pest = "biathlon"
            self.pest_nsamples = 128
            self.pest_seed = 0
            self.qinf = "biathlon"


def get_default_min_confs(args: ExpArgs):
    # return [0.95, 0.98]
    return [0.95]


def get_min_confs(args: ExpArgs):
    # return [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
    return [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]


def get_default_alphas(args: ExpArgs):
    return [5]


def get_default_betas(args: ExpArgs):
    return [1]


def get_all_quantiles(args: ExpArgs):
    return [1, 2, 3, 5] + [i for i in range(10, 100, 20)] + [100]


def get_default_scheduler_cfgs(args: ExpArgs, naggs: int):
    for alpha in get_default_alphas(args):
        for beta in get_default_betas(args):
            cfgs = [(alpha, beta * naggs)]
    return cfgs


def get_scheduler_cfgs(args: ExpArgs, naggs: int):
    quantiles = get_all_quantiles(args)
    default_alphas = get_default_alphas(args)
    default_betas = get_default_betas(args)
    cfgs = []

    # default apha and vary beta
    for alpha in default_alphas:
        for beta in quantiles:
            cfgs.append((alpha, beta * naggs))

    # default beta and vary alpha
    for beta in default_betas:
        for alpha in quantiles[:-1]:
            cfgs.append((alpha, beta * naggs))
    return cfgs


def get_policies(args: ExpArgs) -> List[dict]:
    policies = []
    schedulers = [
        "optimizer",
        "optimizerexpinit",
        "optimizerexpbatch",
        "uniform",
        "uniformexpinit",
        "uniformexpbatch",
        "random",
        "stepgradient",
    ]
    for scheduler in schedulers:
        policies.append({"scheduler": scheduler, "qinf": "biathlon"})
    qinfs = ["biathlon", "prevgradient", "yufei"]
    for qinf in qinfs:
        policies.append({"scheduler": "optimizer", "qinf": qinf})
        policies.append({"scheduler": "stepgradient", "qinf": qinf})
    policies.append({"scheduler": "optimizer", "qinf": "yufei", "pest": "MC"})
    policies.append({"scheduler": "stepgradient", "qinf": "yufei", "pest": "MC"})
    policies.append({"scheduler": "optimizer", "qinf": "prevgradient", "pest": "MC"})
    policies.append({"scheduler": "stepgradient", "qinf": "prevgradient", "pest": "MC"})
    return policies


def list_to_option_str(values: list):
    return " ".join([f"{v}" for v in values])


def get_shared_cmd(args: ExpArgs, task_name: str, model: str):
    shared_cmd = f"{args.interpreter} run.py --example {task_name} \
                    --task {args.task_home}/{task_name} --model {model} \
                    --nparts {args.nparts} --ncores {args.ncores} \
                    --loading_mode {args.loading_mode} --seed {args.seed}"
    return shared_cmd


def get_shared_path(args: ExpArgs, task_name: str, model: str):
    shared_path = os.path.join(EXP_HOME, args.task_home, task_name, f"seed-{args.seed}")
    return shared_path


def run_ingest(args: ExpArgs, task_name: str, model: str):
    shared_cmd = get_shared_cmd(args, task_name, model)
    ingest_cmd = f"{shared_cmd} --stage ingest"
    if args.try_execution:
        print(ingest_cmd)
    else:
        os.system(ingest_cmd)


def run_prepare(args: ExpArgs, task_name: str, model: str):
    shared_cmd = get_shared_cmd(args, task_name, model)
    prepare_cmd = f"{shared_cmd} --stage prepare"
    if args.skip_dataset:
        prepare_cmd = f"{prepare_cmd} --skip_dataset"
    if args.try_execution:
        print(prepare_cmd)
    else:
        os.system(prepare_cmd)


def run_training(args: ExpArgs, task_name: str, model: str):
    shared_cmd = get_shared_cmd(args, task_name, model)
    training_cmd = f"{shared_cmd} --stage train"
    if args.try_execution:
        print(training_cmd)
    else:
        os.system(training_cmd)


def get_base_cmd(args: ExpArgs, task_name: str, model: str):
    shared_cmd = get_shared_cmd(args, task_name, model)
    base_cmd = f"{shared_cmd} --ncfgs {args.ncfgs} \
                --offline_nreqs {args.offline_nreqs}"
    return base_cmd


def get_offline_path(args: ExpArgs, task_name: str, model: str):
    shared_path = get_shared_path(args, task_name, model)
    offline_path = os.path.join(
        shared_path,
        "offline",
        model,
        f"ncores-{args.ncores}",
        f"ldnthreads-{args.loading_mode}",
        f"nparts-{args.nparts}",
        f"ncfgs-{args.ncfgs}",
        f"nreqs-{args.offline_nreqs}",
        "model",
    )
    # qcm_path = os.path.join(offline_path, 'xip_qcm.pkl')
    return offline_path


def run_offline(args: ExpArgs, task_name: str, model: str):
    base_cmd = get_base_cmd(args, task_name, model)
    offline_cmd = f"{base_cmd} --stage offline"
    if args.nocache:
        offline_cmd = f"{offline_cmd} --clear_cache"
    if args.try_execution:
        print(offline_cmd)
    else:
        os.system(offline_cmd)


def get_baseline_path(args: ExpArgs, task_name: str, model: str) -> str:
    shared_path = get_shared_path(args, task_name, model)
    baseline_path = os.path.join(
        shared_path,
        "online",
        model,
        f"ncores-{args.ncores}",
        f"ldnthreads-{args.loading_mode}",
        f"nparts-{args.nparts}",
        "exact",
    )
    return baseline_path


def run_baseline(args: ExpArgs, task_name: str, model: str):
    base_cmd = get_base_cmd(args, task_name, model)
    baseline_cmd = f"{base_cmd} --stage online --exact"
    baseline_path = get_baseline_path(args, task_name, model)
    evals_path = os.path.join(baseline_path, "evals_exact.json")
    if args.nocache or (not os.path.exists(evals_path)):
        if args.try_execution:
            print(baseline_cmd)
        else:
            os.system(baseline_cmd)


def get_ralf_path(args: ExpArgs, task_name: str, model: str) -> str:
    assert args.loading_mode >= 3000
    return get_baseline_path(args, task_name, model)


def run_ralf(args: ExpArgs, task_name: str, model: str):
    assert args.loading_mode >= 3000
    base_cmd = get_base_cmd(args, task_name, model)
    ralf_cmd = f"{base_cmd} --stage online --exact --ralf_budget {args.ralf_budget}"

    if not args.nocache:
        ralf_path = get_baseline_path(args, task_name, model)
        if os.path.exists(ralf_path):
            for file in os.listdir(ralf_path):
                if file.startswith("evals_ralf_"):
                    ralf_budgets = (
                        file.replace("evals_ralf_", "").replace(".json", "").split("_")
                    )
                    if float(ralf_budgets[0]) == args.ralf_budget:
                        print(f"skip {os.path.join(ralf_path, file)}")
                        return None
    if args.try_execution:
        print(ralf_cmd)
    else:
        os.system(ralf_cmd)


def get_biathlon_cmd(
    args: ExpArgs,
    task_name: str,
    model: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
    min_conf: float,
):
    shared_cmd = get_base_cmd(args, task_name, model)
    online_cmd = f"{shared_cmd} --stage online"

    pest_qinf_opts = f"--pest {args.pest} --pest_constraint error --pest_nsamples {args.pest_nsamples} --pest_seed {args.pest_seed} --qinf {args.qinf}"
    scheduler_opts = f"--scheduler {args.scheduler} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch}"
    acc_opts = f"--max_error {max_error} --min_conf {min_conf}"
    bs_opts = f"--bs_type {args.bs_type} --bs_nresamples {args.bs_nresamples} --bs_nthreads {args.bs_nthreads}"
    if not args.bs_feature_correction:
        bs_opts = f"{bs_opts} --bs_feature_correction"
    if not args.bs_bias_correction:
        bs_opts = f"{bs_opts} --bs_bias_correction"
    if not args.bs_for_var_std:
        bs_opts = f"{bs_opts} --bs_for_var_std"
    biathlon_cmd = (
        f"{online_cmd} {pest_qinf_opts} {scheduler_opts} {acc_opts} {bs_opts}"
    )

    return biathlon_cmd


def get_biathlon_path(
    args: ExpArgs,
    task_name: str,
    model: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
    min_conf: float,
):
    shared_path = get_shared_path(args, task_name, model)
    pest_constraint = "error"
    biathlon_path = os.path.join(
        shared_path,
        "online",
        model,
        f"ncores-{args.ncores}",
        f"ldnthreads-{args.loading_mode}",
        f"nparts-{args.nparts}",
        f"ncfgs-{args.ncfgs}",
        f"pest-{pest_constraint}-{args.pest}-{args.pest_nsamples}-{args.pest_seed}",
        f"qinf-{args.qinf}",
        f"scheduler-{args.scheduler}-{scheduler_init}-{scheduler_batch}",
    )
    if args.bs_type != "fstd":
        biathlon_path = os.path.join(
            biathlon_path,
            f"bs-{args.bs_type}-{args.bs_nresamples}-{args.bs_nthreads}-{args.bs_feature_correction}-{args.bs_bias_correction}-{args.bs_for_var_std}",
        )
    return biathlon_path


def run_biathlon(
    args: ExpArgs,
    task_name: str,
    model: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
    min_conf: float,
):
    biathlon_cmd = get_biathlon_cmd(
        args, task_name, model, scheduler_init, scheduler_batch, max_error, min_conf
    )
    biathlon_path = get_biathlon_path(
        args, task_name, model, scheduler_init, scheduler_batch, max_error, min_conf
    )
    evals_file = f"evals_conf-0.05-{max_error}-{min_conf}-60.0-2048.0-1000.json"
    evals_path = os.path.join(biathlon_path, evals_file)
    if args.nocache or (not os.path.exists(evals_path)):
        if args.try_execution:
            print(biathlon_cmd)
        else:
            os.system(biathlon_cmd)
    else:
        print(f"skip {evals_path}")


def run_tempbiathlon(
    args: ExpArgs,
    task_name: str,
    model: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
    min_conf: float,
):
    biathlon_cmd = get_biathlon_cmd(
        args, task_name, model, scheduler_init, scheduler_batch, max_error, min_conf
    )
    biathlon_path = get_biathlon_path(
        args, task_name, model, scheduler_init, scheduler_batch, max_error, min_conf
    )
    biathlon_cmd = biathlon_cmd.replace("--stage online", "--stage temponline")
    biathlon_path = biathlon_path.replace("/online/", "/temponline/")
    evals_file = f"evals_conf-0.05-{max_error}-{min_conf}-60.0-2048.0-1000.json"
    evals_path = os.path.join(biathlon_path, evals_file)
    if args.nocache or (not os.path.exists(evals_path)):
        if args.try_execution:
            print(biathlon_cmd)
        else:
            os.system(biathlon_cmd)
    else:
        print(f"skip {evals_path}")


def run_profile(
    args: ExpArgs,
    task_name: str,
    model: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
    min_conf: float,
):

    biathlon_cmd = get_biathlon_cmd(
        args, task_name, model, scheduler_init, scheduler_batch, max_error, min_conf
    )

    offline_path = get_offline_path(args, task_name, model)
    profile_dir = os.path.join(offline_path, "..", "profile")
    os.makedirs(profile_dir, exist_ok=True)

    profile_tag = (
        f"biathlon_default_{scheduler_init}_{scheduler_batch}_{max_error}_{min_conf}"
    )

    profling_opts = f"-m cProfile -s cumtime -o {profile_dir}/{profile_tag}.pstats"
    # add profiling opts after args.interpreter
    cmd = biathlon_cmd.replace(args.interpreter, f"{args.interpreter} {profling_opts}")
    if args.try_execution:
        print(cmd)
    else:
        os.system(cmd)

    cmd = f"gprof2dot -f pstats {profile_dir}/{profile_tag}.pstats | dot -Tsvg -o {profile_dir}/{profile_tag}.svg"
    if args.try_execution:
        print(cmd)
    else:
        os.system(cmd)

    print(f"profiling results in {profile_dir}/{profile_tag}.svg")


def run_verbose(
    args: ExpArgs,
    task_name: str,
    model: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
    min_conf: float,
):
    nreqs = 1
    biathlon_cmd = get_biathlon_cmd(
        args, task_name, model, scheduler_init, scheduler_batch, max_error, min_conf
    )
    cmd = f"{biathlon_cmd} --verbose --nreqs {nreqs}"
    if args.try_execution:
        print(cmd)
    else:
        os.system(cmd)


def run_pipeline(
    args: ExpArgs,
    task_name: str,
    agg_qids: str,
    default_max_errors: list[float],
    max_errors: list[float],
    default_only: bool = False,
):
    naggs = len(agg_qids.split(" "))
    model = args.model

    if args.phase == "ingest":
        run_ingest(args, task_name, model)
    elif args.phase == "prepare":
        run_prepare(args, task_name, model)
    elif args.phase == "training":
        run_training(args, task_name, model)
    elif args.phase == "offline":
        run_offline(args, task_name, model)
    elif args.phase == "setup":
        run_ingest(args, task_name, model)
        run_prepare(args, task_name, model)
        run_training(args, task_name, model)
        run_offline(args, task_name, model)
    elif args.phase == "baseline":
        run_baseline(args, task_name, model)
    elif args.phase == "ralf":
        run_ralf(args, task_name, model)
    elif args.phase == "ralfs":
        assert args.ralf_budgets is not None
        ralf_budget_cache = args.ralf_budget
        for ralf_budget in args.ralf_budgets:
            args.ralf_budget = ralf_budget
            run_ralf(args, task_name, model)
        args.ralf_budget = ralf_budget_cache
    elif args.phase == "default":
        run_biathlon(
            args,
            task_name,
            model,
            args.default_alpha,
            args.default_beta * naggs,
            args.default_error,
            args.default_conf,
        )
    elif args.phase == "tempdefault":
        run_tempbiathlon(
            args,
            task_name,
            model,
            args.default_alpha,
            args.default_beta * naggs,
            args.default_error,
            args.default_conf,
        )
    elif args.phase == "test":
        run_tempbiathlon(
            args,
            task_name,
            model,
            args.test_alpha,
            args.test_beta * naggs,
            args.test_error,
            args.test_conf,
        )
    elif args.phase == "warmup":
        run_biathlon(args, task_name, model, 1000, 1000 * naggs, 0.0, 1.1)
    elif args.phase == "profile":
        run_profile(
            args,
            task_name,
            model,
            args.default_alpha,
            args.default_beta * naggs,
            args.default_error,
            args.default_conf,
        )
    elif args.phase == "verbose":
        run_verbose(
            args,
            task_name,
            model,
            args.default_alpha,
            args.default_beta * naggs,
            args.default_error,
            args.default_conf,
        )
    elif args.phase == "biathlon-default":
        for sch_init, sch_batch in get_default_scheduler_cfgs(args, naggs):
            for max_error in default_max_errors:
                for min_conf in get_default_min_confs(args):
                    run_biathlon(
                        args, task_name, model, sch_init, sch_batch, max_error, min_conf
                    )
    elif args.phase == "biathlon-confs":
        for sch_init, sch_batch in get_default_scheduler_cfgs(args, naggs):
            for max_error in default_max_errors:
                for min_conf in get_min_confs(args):
                    run_biathlon(
                        args, task_name, model, sch_init, sch_batch, max_error, min_conf
                    )
    elif args.phase == "biathlon-errors":
        # vary max_error only
        for sch_init, sch_batch in get_default_scheduler_cfgs(args, naggs):
            for max_error in max_errors:
                for min_conf in get_default_min_confs(args):
                    run_biathlon(
                        args, task_name, model, sch_init, sch_batch, max_error, min_conf
                    )
    elif args.phase == "biathlon-alphas":
        for sch_init in get_all_quantiles(args):
            for sch_batch in get_default_betas(args):
                for max_error in default_max_errors:
                    for min_conf in get_default_min_confs(args):
                        run_biathlon(
                            args,
                            task_name,
                            model,
                            sch_init,
                            sch_batch,
                            max_error,
                            min_conf,
                        )
    elif args.phase == "biathlon-betas":
        for sch_init in get_default_alphas(args):
            for sch_batch in get_all_quantiles(args):
                for max_error in default_max_errors:
                    for min_conf in get_default_min_confs(args):
                        run_biathlon(
                            args,
                            task_name,
                            model,
                            sch_init,
                            sch_batch,
                            max_error,
                            min_conf,
                        )
    elif args.phase == "biathlon-alphasbetas":
        for sch_init, sch_batch in get_scheduler_cfgs(args, naggs):
            for max_error in default_max_errors:
                for min_conf in get_default_min_confs(args):
                    run_biathlon(
                        args, task_name, model, sch_init, sch_batch, max_error, min_conf
                    )
    elif args.phase == "biathlon-policy":
        for sch_init, sch_batch in get_default_scheduler_cfgs(args, naggs):
            for max_error in default_max_errors:
                for min_conf in get_default_min_confs(args):
                    for policy in get_policies(args):
                        dup_args = ExpArgs(args).parse_args()
                        dup_args.scheduler = policy.get("scheduler", args.scheduler)
                        dup_args.qinf = policy.get("qinf", args.qinf)
                        dup_args.pest = policy.get("pest", args.pest)
                        run_biathlon(
                            dup_args,
                            task_name,
                            model,
                            sch_init,
                            sch_batch,
                            max_error,
                            min_conf,
                        )
    elif args.phase == "biathlon-bs_type":
        for sch_init, sch_batch in get_default_scheduler_cfgs(args, naggs):
            for max_error in default_max_errors:
                for min_conf in get_default_min_confs(args):
                    for bs_type in ['fstd', 'descrete', 'allbs']:
                        dup_args = ExpArgs(args).parse_args()
                        dup_args.bs_type = bs_type
                        run_biathlon(
                            dup_args,
                            task_name,
                            model,
                            sch_init,
                            sch_batch,
                            max_error,
                            min_conf,
                        )
    elif args.phase == "biathlon":
        default_cfgs = get_default_scheduler_cfgs(args, naggs)
        cfgs = get_scheduler_cfgs(args, naggs)

        default_min_confs = get_default_min_confs(args)
        min_confs = get_min_confs(args)

        # default only
        for sch_init, sch_batch in default_cfgs:
            for max_error in default_max_errors:
                for min_conf in default_min_confs:
                    run_biathlon(
                        args, task_name, model, sch_init, sch_batch, max_error, min_conf
                    )
        if default_only or args.default_only:
            return

        # vary max_error only
        for sch_init, sch_batch in default_cfgs:
            for max_error in max_errors:
                for min_conf in default_min_confs:
                    run_biathlon(
                        args, task_name, model, sch_init, sch_batch, max_error, min_conf
                    )

        # vary cfgs only
        for sch_init, sch_batch in cfgs:
            for max_error in default_max_errors:
                for min_conf in default_min_confs:
                    run_biathlon(
                        args, task_name, model, sch_init, sch_batch, max_error, min_conf
                    )

        # vary min_conf
        for sch_init, sch_batch in default_cfgs:
            for max_error in default_max_errors:
                for min_conf in min_confs:
                    run_biathlon(
                        args, task_name, model, sch_init, sch_batch, max_error, min_conf
                    )
    else:
        raise ValueError(f"invalid phase {args.phase}")


def run_studentqno(args: ExpArgs, qno: int):
    """
    models = [lgbm, gbm, tfgbm]
    """
    task_name = f"studentqno{qno}"
    agg_qids = list_to_option_str([i for i in range(13)])
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_studentqnotest(args: ExpArgs):
    """
    models = [lgbm, gbm, tfgbm]
    """
    task_name = "studentqnotest"
    agg_qids = list_to_option_str([i for i in range(13)])
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_student(args: ExpArgs):
    """
    models = [lgbm, gbm, tfgbm]
    """
    task_name = "student"
    agg_qids = list_to_option_str([i + 1 for i in range(13)])
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_studentqnov2(args: ExpArgs, task_name: str = "studentqnov2"):
    """
    models = [lgbm, gbm, tfgbm]
    """
    agg_qids = list_to_option_str([i for i in range(13)])
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_performance(args: ExpArgs, task_name: str = "performance18"):
    """
    models = [lgbm, gbm, tfgbm]
    """
    agg_qids = list_to_option_str([i for i in range(13)])
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_machinery(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machinery"
    agg_qids = "0 1 2 3 4 5 6 7"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_machineryralftest(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machineryralftest"
    agg_qids = "0 1 2 3 4 5 6 7"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_machineryralf(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machineryralf"
    agg_qids = "0 1 2 3 4 5 6 7"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_machineryralfmedian(args: ExpArgs, task_name: str):
    agg_qids = "0 1 2 3 4 5 6 7"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_machinerymulti(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machinerymulti"
    agg_qids = "0 1 2 3 4 5 6 7"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_cheaptrips(args: ExpArgs):
    """
    must models = ["xgb"]
    optional models = ["dt", "lgbm", "rf"]
    """
    task_name = "cheaptrips"
    agg_qids = "1 2 3"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_ccfraud(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "xgb", "lgbm", "rf"]
    """
    task_name = "ccfraud"
    agg_qids = "3 4 5 6"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tdfraud(args: ExpArgs, task_name: str = "tdfraud"):
    """
    must models = ["xgb"]
    """
    agg_qids = "1 2 3"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_trips(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "trips"
    agg_qids = "1 2 3"
    default_max_errors = [0.1, 0.5, 1.0]
    max_errors = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralf(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralf"
    agg_qids = "1 2"
    default_max_errors = [1.0, 2.0, 2.68, 5.35]
    max_errors = [0.1, 0.5, 1.0, 2.0, 2.68, 4.0, 5.35, 6.0, 8.0, 10.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralfv2(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralfv2"
    agg_qids = "1 2"
    # default_max_errors = [0.75, 1.0, 1.5, 2.0]
    # max_errors = [0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    default_max_errors = [1.5]
    max_errors = [0.1875, 0.375, 0.75, 1.5, 3.0, 6.0, 9.0, 12.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralfv3(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralfv3"
    agg_qids = "1 2"
    # default_max_errors = [0.7, 1.4, 2.8]
    default_max_errors = [1.4]
    max_errors = [0.175, 0.35, 0.7, 1.0, 1.4, 2.8, 5.6, 11.2, 22.4]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralfv2median(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralfv2median"
    agg_qids = "1 2"
    default_max_errors = [1.5]
    max_errors = [0.1875, 0.375, 0.75, 1.5, 3.0, 6.0, 9.0, 12.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralfv2simmedian(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralfv2simmedian"
    agg_qids = "1 2"
    default_max_errors = [1.4]
    max_errors = [0.175, 0.35, 0.7, 1.0, 1.4, 2.8, 5.6, 11.2, 22.4]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralfv3median(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralfv3median"
    agg_qids = "1 2"
    default_max_errors = [1.4]
    max_errors = [0.175, 0.35, 0.7, 1.0, 1.4, 2.8, 5.6, 11.2, 22.4]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralfv3simmedian(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralfv3simmedian"
    agg_qids = "1 2"
    default_max_errors = [1.4]
    max_errors = [0.175, 0.35, 0.7, 1.0, 1.4, 2.8, 5.6, 11.2, 22.4]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralftest(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralftest"
    agg_qids = "1 2"
    # default_max_errors = [0.5, 1.0, 1.5]
    default_max_errors = [1.0]
    max_errors = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsralf2h(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsralf2h"
    agg_qids = "1 2"
    default_max_errors = [0.5, 1.0, 2.0]
    max_errors = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsfeast(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsfeast"
    agg_qids = "1 2"
    default_max_errors = [0.1, 0.5, 1.0]
    max_errors = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_batterytest(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "batterytest"
    agg_qids = "0 1 2 3 4"
    # default_max_errors = [120.0, 300.0]
    default_max_errors = [300.0]
    max_errors = [60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 3600.0, 7200.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_battery(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "battery"
    agg_qids = "0 1 2 3 4"
    default_max_errors = [120.0, 300.0]
    max_errors = [60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 3600.0, 7200.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_batteryv2(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "batteryv2"
    agg_qids = "0 1 2 3 4"
    default_max_errors = [186.7]
    max_errors = [
        30.0,
        46.675,
        60.0,
        93.35,
        120,
        186.7,
        300.0,
        373.4,
        600.0,
        746.8,
        1493.6,
        1500.0,
        1800.0,
        2400.0,
        2987.2,
        3000.0,
        3600.0,
        4800.0,
        5974.4,
        7200.0,
        11948.8,
        23897.6,
    ]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_batteryv2median(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "batteryv2median"
    agg_qids = "0 1 2 3 4"
    default_max_errors = [186.7]
    max_errors = [
        30.0,
        46.675,
        60.0,
        93.35,
        120,
        186.7,
        300.0,
        373.4,
        600.0,
        746.8,
        900.0,
        1200.0,
        1500.0,
        1800.0,
        2400.0,
        3000.0,
        3600.0,
        4800.0,
        7200.0,
    ]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_batteryv2simmedian(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "batteryv2simmedian"
    agg_qids = "0 1 2 3 4"
    default_max_errors = [186.7]
    max_errors = [
        30.0,
        46.675,
        60.0,
        93.35,
        120,
        186.7,
        300.0,
        373.4,
        600.0,
        746.8,
        900.0,
        1200.0,
        1500.0,
        1800.0,
        2400.0,
        3000.0,
        3600.0,
        4800.0,
        7200.0,
    ]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_turbofan(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "turbofan"
    naggs = 9
    agg_qids = list_to_option_str([i for i in range(naggs)])
    default_max_errors = [4.88]
    max_errors = [0.61, 1.22, 2.44, 4.88, 9.76, 19.52, 39.04, 78.08]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_turbofanmedian(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "turbofanmedian"
    naggs = 9
    agg_qids = list_to_option_str([i for i in range(naggs)])
    default_max_errors = [4.88]
    max_errors = [0.61, 1.22, 2.44, 4.88, 9.76, 19.52, 39.04, 78.08]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_turbofansimmedian(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "turbofansimmedian"
    naggs = 9
    agg_qids = list_to_option_str([i for i in range(naggs)])
    default_max_errors = [4.88]
    max_errors = [0.61, 1.22, 2.44, 4.88, 9.76, 19.52, 39.04, 78.08]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_turbofanall(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "turbofanall"
    naggs = 44
    agg_qids = list_to_option_str([i for i in range(naggs)])
    default_max_errors = [1.0, 3.0, 6.0]
    max_errors = [1.0, 3.0, 6.0, 10.0, 20.0, 50.0, 80.0, 100.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_cheaptripsfeast(args: ExpArgs):
    """
    must models = ["xgb"]
    optional models = ["dt", "lgbm", "rf"]
    """
    task_name = "cheaptripsfeast"
    agg_qids = "1 2"
    default_max_errors = [0.0]
    max_errors = [0.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tick_v1(args: ExpArgs):
    """
    models = ["lr", "dt", "rf"]
    """
    task_name = "tick"
    agg_qids = "1"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tick_v2(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickv2"
    agg_qids = "6"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tickralf(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickralf"
    agg_qids = "6"
    default_max_errors = [0.01, 0.02, 0.04, 0.05]
    max_errors = [0.001, 0.01, 0.02, 0.04, 0.05, 0.1]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tickralftest(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickralftest"
    agg_qids = "6"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tickralfv2(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickralfv2"
    agg_qids = "6"
    default_max_errors = [0.04]
    max_errors = [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.32, 500.0, 10000000000.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tickralfv2median(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickralfv2median"
    agg_qids = "6"
    default_max_errors = [0.04]
    max_errors = [0.005, 0.01, 0.02, 0.04, 0.06, 0.12, 500.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tickralfv2simmedian(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickralfv2simmedian"
    agg_qids = "6"
    default_max_errors = [0.04]
    max_errors = [0.005, 0.01, 0.02, 0.04, 0.06, 0.12, 500.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tickralfv2test(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickralfv2test"
    agg_qids = "6"
    default_max_errors = [0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_studentqno18_vary_nf(args: ExpArgs, nf: int):
    """
    models = [lgbm, gbm, tfgbm]
    """
    task_name = f"studentqno18nf{nf}"
    agg_qids = list_to_option_str([i for i in range(nf)])
    run_pipeline(args, task_name, agg_qids, [0.0], [0.0], default_only=True)


def run_machineryralf_vary_nf(args: ExpArgs, nf: int, fixed: bool = False):
    """
    must models = ["mlp", "svm", "knn"]
    """
    if fixed:
        task_name = f"machineryralfxf{nf}"
    else:
        task_name = f"machineryralfnf{nf}"
    agg_qids = list_to_option_str([i for i in range(nf)])
    run_pipeline(args, task_name, agg_qids, [0.0], [0.0], default_only=True)


def run_machinery_vary_nf(args: ExpArgs, nf: int, fixed: bool = False):
    """
    must models = ["mlp", "svm", "knn"]
    """
    if fixed:
        task_name = f"machineryxf{nf}"
    else:
        task_name = f"machinerynf{nf}"
    agg_qids = list_to_option_str([i for i in range(nf)])
    run_pipeline(args, task_name, agg_qids, [0.0], [0.0], default_only=True)


def run_machinerymulti_vary_nf(args: ExpArgs, nf: int, fixed: bool = False):
    """
    must models = ["mlp", "svm", "knn"]
    """
    if fixed:
        task_name = f"machinerymultixf{nf}"
    else:
        task_name = f"machinerymultinf{nf}"
    agg_qids = list_to_option_str([i for i in range(nf)])
    run_pipeline(args, task_name, agg_qids, [0.0], [0.0], default_only=True)


def run_tick_vary_nmonths(args: ExpArgs, nmonths: int):
    """
    models = ["lr", "dt", "rf"]
    """
    task_name = f"tickvaryNM{nmonths}"
    agg_qids = "6"
    run_pipeline(
        args, task_name, agg_qids, [0.01, 0.05], [0.01, 0.05], default_only=True
    )


def run_tripsfeast_vary_window_size(args: ExpArgs, nmonths: int):
    """
    models = ["lgbm", "xgb", "dt", "rf"]
    """
    task_name = f"tripsfeastw{nmonths}"
    agg_qids = "1 2"
    run_pipeline(args, task_name, agg_qids, [1.0, 1.66], [1.0, 1.66], default_only=True)


def run_tick_price(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickvaryNM1"
    agg_qids = "6"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tick_price_middle(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickvaryNM8"
    agg_qids = "6"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_extreme_tick_price(args: ExpArgs):
    """
    must models = ["lr"]
    """
    task_name = "tickvaryNM8"
    agg_qids = "6"
    default_max_errors = [0.0001, 0.0005]
    max_errors = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    run_pipeline(
        args, task_name, agg_qids, default_max_errors, max_errors, default_only=True
    )


if __name__ == "__main__":
    args = ExpArgs().parse_args()
    if args.exp == "tripsralf":
        run_tripsralf(args)
    elif args.exp == "tripsralftest":
        run_tripsralftest(args)
    elif args.exp == "tripsralf2h":
        run_tripsralf2h(args)
    elif args.exp == "tripsralfv2":
        run_tripsralfv2(args)
    elif args.exp == "tripsralfv3":
        run_tripsralfv3(args)
    elif args.exp == "tripsralfv2median":
        run_tripsralfv2median(args)
    elif args.exp == "tripsralfv3median":
        run_tripsralfv3median(args)
    elif args.exp == "tripsralfv2simmedian":
        run_tripsralfv2simmedian(args)
    elif args.exp == "tripsralfv3simmedian":
        run_tripsralfv3simmedian(args)
    elif args.exp == "trips":
        run_trips(args)
    elif args.exp == "tick-v1":
        run_tick_v1(args)
    elif args.exp == "tick-v2":
        run_tick_v2(args)
    elif args.exp == "tickralf":
        run_tickralf(args)
    elif args.exp == "tickralftest":
        run_tickralftest(args)
    elif args.exp == "tickralfv2":
        run_tickralfv2(args)
    elif args.exp == "tickralfv2median":
        run_tickralfv2median(args)
    elif args.exp == "tickralfv2simmedian":
        run_tickralfv2simmedian(args)
    elif args.exp == "tickralfv2test":
        run_tickralfv2test(args)
    elif args.exp == "cheaptrips":
        run_cheaptrips(args)
    elif args.exp == "machinery":
        run_machinery(args)
    elif args.exp == "machineryralf":
        run_machineryralf(args)
    elif args.exp == "machineryralftest":
        run_machineryralftest(args)
    elif (
        args.exp.startswith("machineryralfe2emedian")
        or args.exp.startswith("machineryralfdirectmedian")
        or args.exp.startswith("machineryralfsimmedian")
        or args.exp.startswith("machineryralfmedian")
    ):
        run_machineryralfmedian(args, task_name=args.exp)
    elif args.exp == "ccfraud":
        run_ccfraud(args)
    elif args.exp == "tripsfeast":
        run_tripsfeast(args)
    elif args.exp == "cheaptripsfeast":
        run_cheaptripsfeast(args)
    elif args.exp == "machinerymulti":
        run_machinerymulti(args)
    elif args.exp == "studentqnotest":
        run_studentqnotest(args)
    elif args.exp in StudentQNo18VaryNF:
        nf = int(args.exp[len("studentqno18nf") :])
        run_studentqno18_vary_nf(args, nf)
    elif args.exp in MachineryVaryNF:
        nf = int(args.exp[len("machinerynf") :])
        run_machinery_vary_nf(args, nf)
    elif args.exp in MachineryMultiVaryNF:
        nf = int(args.exp[len("machinerymultinf") :])
        run_machinerymulti_vary_nf(args, nf)
    elif args.exp in MachineryRalfVaryNF:
        nf = int(args.exp[len("machineryralfnf") :])
        run_machineryralf_vary_nf(args, nf)
    elif args.exp in MachineryVaryXNF:
        nf = int(args.exp[len("machineryxf") :])
        run_machinery_vary_nf(args, nf, fixed=True)
    elif args.exp in MachineryMultiVaryXNF:
        nf = int(args.exp[len("machinerymultixf") :])
        run_machinerymulti_vary_nf(args, nf, fixed=True)
    elif args.exp in TickVaryNMonths:
        nmonths = int(args.exp[len("tickvaryNM") :])
        run_tick_vary_nmonths(args, nmonths)
    elif args.exp in TripsFeastVaryWindow:
        nmonths = int(args.exp[len("tripsfeastw") :])
        run_tripsfeast_vary_window_size(args, nmonths)
    elif args.exp == "tickprice":
        run_tick_price(args)
    elif args.exp.startswith("tdfraud"):
        run_tdfraud(args, task_name=args.exp)
    elif args.exp == "tickpricemiddle":
        run_tick_price_middle(args)
    elif args.exp == "extremetickprice":
        run_extreme_tick_price(args)
    elif args.exp == "batterytest":
        run_batterytest(args)
    elif args.exp == "battery":
        run_battery(args)
    elif args.exp == "batteryv2":
        run_batteryv2(args)
    elif args.exp == "batteryv2median":
        run_batteryv2median(args)
    elif args.exp == "batteryv2simmedian":
        run_batteryv2simmedian(args)
    elif args.exp == "turbofan":
        run_turbofan(args)
    elif args.exp == "turbofanmedian":
        run_turbofanmedian(args)
    elif args.exp == "turbofansimmedian":
        run_turbofansimmedian(args)
    elif args.exp == "turbofanall":
        run_turbofanall(args)
    elif args.exp == "student":
        run_student(args)
    elif args.exp.startswith("studentqnov2"):
        run_studentqnov2(args, task_name=args.exp)
    elif args.exp.startswith("studentqno"):
        qno = int(args.exp[len("studentqno") :])
        run_studentqno(args, qno)
    elif args.exp.startswith("performance"):
        run_performance(args, task_name=args.exp)
    elif args.exp == "Trip-Fare":
        run_tripsralfv3(args)
    elif args.exp == "Tick-Price":
        run_tickralfv2(args)
    elif args.exp == "Battery":
        run_batteryv2(args)
    elif args.exp == "Turbofan":
        run_turbofan(args)
    elif args.exp == "Fraud-Detection":
        run_tdfraud(args, task_name="tdfraudralf2d")
    elif args.exp == "Bearing-Imbalance":
        run_machineryralf(args)
    elif args.exp == "Student-QA":
        run_studentqnov2(args, task_name="studentqnov2subset")
    elif args.exp == "Trip-Fare-Median":
        run_tripsralfv3median(args)
    elif args.exp == "Tick-Price-Median":
        run_tickralfv2median(args)
    elif args.exp == "Battery-Median":
        run_batteryv2median(args)
    elif args.exp == "Turbofan-Median":
        run_turbofanmedian(args)
    elif args.exp == "Fraud-Detection-Median":
        run_tdfraud(args, task_name="tdfraudralf2dv2median")
    elif args.exp == "Bearing-Imbalance-Median":
        run_machineryralfmedian(args)
    elif args.exp == "Student-QA-Median":
        run_studentqnov2(args, task_name="studentqnov2subsetmedian")
    else:
        raise ValueError(f"invalid exp {args.exp}")
