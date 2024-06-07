import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import os
import numpy as np
import math
import json
from matplotlib.transforms import Bbox
from typing import List
from tap import Tap
from sklearn import metrics
from tqdm import tqdm

from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS


class VLDBArgs(Tap):
    data_dir: str = "/home/ckchang/.cache/biathlon/vldb2024/servers/2024042117"
    out_dir: str = "./cache"
    filename: str = None
    debug: bool = False

    def process_args(self) -> None:
        if self.filename is None:
            name = os.path.basename(self.data_dir)
            self.filename = f"{name}.csv"


def path_parser(args: VLDBArgs, path: str) -> dict:
    """
    Three types of paths:
    1. {data_dir}/{server}/{task_name}/seed-{seed}/{stage}/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}/{system}/evals_exact.json
    2. {data_dir}/{server}/{task_name}/seed-{seed}/{stage}/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}/ncfgs-{ncfgs}/pest-{pest_constraint}-{pest}-{pest_nsamples}-{pest_seed}/qinf-{qinf}/scheduler-{policy}-{scheduler_init}-{scheduler_batch}/evals_{condition_type}-{max_relerror}-{max_error}-{min_conf}-{max_time}-{max_time}-{max_rounds}.json
    3. {data_dir}/{server}/{task_name}/seed-{seed}/{stage}/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}/{system}/evals_ralf_{ralf_budget}_..._{ralf_budget}.json
    extract all the settings in the path
    """
    assert path.startswith(args.data_dir), f"{path} does not start with {args.data_dir}"

    settings = {}

    dirname = os.path.dirname(path)
    dirname = dirname.replace(args.data_dir, '').lstrip('/')

    components = dirname.split('/')
    components = [args.data_dir] + components

    settings["server"] = components[1]
    settings["task_name"] = components[2]
    settings["seed"] = components[3].split('-')[1]

    assert components[4] == 'online'
    settings["stage"] = components[4]

    settings["model"] = components[5]
    settings["ncores"] = components[6].split('-')[1]
    settings["loading_mode"] = components[7].split('-')[1]
    settings["nparts"] = components[8].split('-')[1]

    file = os.path.basename(path)
    file = file.replace('evals_', '').replace('.json', '')

    if file == 'exact':
        settings["system"] = components[9]
        settings["max_error"] = 0.0
        settings["min_conf"] = 1.0
    elif file.startswith('ralf'):
        settings["system"] = "ralf"
        ralf_budgets = file.split('_')[1:]
        # all ralf budgets should be same
        assert all([ralf_budgets[0] == ralf_budget for ralf_budget in ralf_budgets]), f"ralf budgets are not same: {ralf_budgets}"
        settings["ralf_budget"] = ralf_budgets[0]
    elif file.startswith('conf'):
        settings["system"] = "biathlon"
        settings["ncfgs"] = components[9].split('-')[1]

        assert components[10].startswith("pest-"), f"components[10] = {components[10]}"
        pest_constraint, pest, pest_nsamples, pest_seed = components[10].split('-')[1:]
        settings["pest_constraint"] = pest_constraint
        settings["pest"] = pest
        settings["pest_nsamples"] = pest_nsamples
        settings["pest_seed"] = pest_seed

        assert components[11].startswith("qinf-")
        settings["qinf"] = components[11].split('-')[1]

        assert components[12].startswith("scheduler-")
        policy, scheduler_init, scheduler_batch = components[12].split('-')[1:]
        settings["policy"] = policy
        settings["scheduler_init"] = scheduler_init
        settings["scheduler_batch"] = scheduler_batch

        if len(components) > 13:
            assert components[13].startswith("bs-")
            bs_type, bs_nresamples, bs_nthreads, bs_feature_correction, bs_bias_correction, bs_for_var_std = components[13].split('-')[1:]
            settings["bs_type"] = bs_type
            settings["bs_nresamples"] = bs_nresamples
            settings["bs_nthreads"] = bs_nthreads
            settings["bs_feature_correction"] = bs_feature_correction
            settings["bs_bias_correction"] = bs_bias_correction
            settings["bs_for_var_std"] = bs_for_var_std
        else:
            settings["bs_type"] = "fstd"
            settings["bs_nresamples"] = 100
            settings["bs_nthreads"] = 1
            settings["bs_feature_correction"] = True
            settings["bs_bias_correction"] = True
            settings["bs_for_var_std"] = True


        condition_type, max_relerror, max_error, min_conf, max_time, max_memory, max_rounds = file.split('-')
        settings["condition_type"] = condition_type
        settings["max_relerror"] = max_relerror
        settings["max_error"] = max_error
        settings["min_conf"] = min_conf
        settings["max_time"] = max_time
        settings["max_memory"] = max_memory
        settings["max_rounds"] = max_rounds
    else:
        raise ValueError(f"Unknown file: {file}")

    return settings


def get_baseline_key(args: VLDBArgs, settings: dict):
    key = f"{args.data_dir}/{settings['server']}/{settings['task_name']}/seed-{settings['seed']}/{settings['stage']}/{settings['model']}/ncores-{settings['ncores']}/ldnthreads-0/nparts-{settings['nparts']}/exact/final_df_exact.csv"
    return key


def aggregate_data(args: VLDBArgs, data: pd.DataFrame) -> pd.DataFrame:
    dropped_cols = ['server']
    df = data.drop(columns=dropped_cols, errors='ignore')
    keys = ['task_name', 'model', 'stage',
            'ncores', 'loading_mode', 'nparts', 'ncfgs',
            'system', 'pest', 'pest_constraint',
            'pest_nsamples', 'pest_seed',
            'qinf', 'ralf_budget',
            'policy', 'scheduler_init', 'scheduler_batch',
            'condition_type', 'max_relerror',
            'max_error', 'min_conf',
            'max_time', 'max_memory', 'max_rounds', 
            "bs_type", "bs_nthreads", "bs_nresamples",
            "bs_feature_correction", "bs_bias_correction",
            "bs_for_var_std"]
    # deduplicate by the keys + ['seed'], keep the first
    print(f'total number of rows: {len(df)}')
    df = df.groupby(keys + ['seed'], dropna=False).first().reset_index()
    print(f'total number of rows: {len(df)}')
    df_grp = df.groupby(keys, dropna=False)

    list_cols = ["avg_sample_query", "avg_qtime_query"]
    non_list_cols = [col for col in df.columns if col not in list_cols + keys]
    # for columns except list_cols, take the mean;
    # for list_cols, take the mean of each element
    # print(f'non_list_cols: {non_list_cols}')
    df_non_list = df_grp[non_list_cols].mean().reset_index()
    df_list = []
    for col in list_cols:
        if col in df.columns:
            df_list.append(df_grp[col].apply(lambda x: np.mean(x.tolist(), axis=0).tolist()).reset_index())
    df = df_non_list
    for tmp in df_list:
        df = pd.merge(df, tmp, on=keys)
    print(f'total number of rows: {len(df)}')

    return df


if __name__ == "__main__":
    args = VLDBArgs().parse_args()
    data_dir = args.data_dir

    # final all evals_{xxx}.json and final_df_{xxxx}.csv in data_dir (recursively)
    path2evals_dict = {'exact': [], 'biathlon': [], 'ralf': []}
    path2dfs_dict = {'exact': [], 'biathlon': [], 'ralf': []}
    bsl_dfs = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.startswith('evals_') and file.endswith('.json'):
                df_file = file.replace('evals_', 'final_df_').replace('.json', '.csv')
                assert os.path.exists(os.path.join(root, df_file)), f"{os.path.join(root, df_file)} not found"

                if file.startswith('evals_exact'):
                    path2evals_dict['exact'].append(os.path.join(root, file))
                    path2dfs_dict['exact'].append(os.path.join(root, df_file))
                elif file.startswith('evals_ralf'):
                    path2evals_dict['ralf'].append(os.path.join(root, file))
                    path2dfs_dict['ralf'].append(os.path.join(root, df_file))
                elif file.startswith('evals_conf'):
                    path2evals_dict['biathlon'].append(os.path.join(root, file))
                    path2dfs_dict['biathlon'].append(os.path.join(root, df_file))
                else:
                    raise ValueError(f"Unknown file: {file}")

                if df_file == 'final_df_exact.csv':
                    key = os.path.join(root, df_file)
                    bsl_dfs[key] = pd.read_csv(key)

    path2evals = path2evals_dict['exact'] + path2evals_dict['biathlon'] + path2evals_dict['ralf']
    path2dfs = path2dfs_dict['exact'] + path2dfs_dict['biathlon'] + path2dfs_dict['ralf']

    data = []
    max_errors = {}
    for path2eval, path2df in tqdm(zip(path2evals, path2dfs),
                                   total=len(path2evals),
                                   desc="Processing"):
        settings = path_parser(args, path2eval)
        bsl_key = get_baseline_key(args, settings)
        with open(path2eval, 'r') as f:
            evals = json.load(f)
        df = pd.read_csv(path2df)
        bsl_df = bsl_dfs[bsl_key]
        bsl_avg_lat = bsl_df['latency'].mean()

        item = {**settings}
        item['system_cost'] = evals['avg_ppl_time']
        if item['system'] == 'ralf':
            item['avg_latency'] = evals['avg_ppl_time'] - evals['avg_query_time']
            item['BD:AFC'] = 1e-2
        else:
            item['avg_latency'] = evals['avg_ppl_time']
            item['BD:AFC'] = evals['avg_query_time']

        item['BD:AMI'] = evals['avg_pred_time']
        item['BD:Sobol'] = evals['avg_scheduler_time']
        item['avg_nrounds'] = evals['avg_nrounds']
        item['sampling_rate'] = evals['avg_sample']

        item['speedup'] = bsl_avg_lat / item['avg_latency']

        agg_fnames = [col for col in bsl_df.columns if col.startswith('AGG_')]
        agg_ops = [fname.split('_')[-1] for fname in agg_fnames]
        item['naggs'] = len(set(agg_ops))

        item['avg_pred_conf'] = np.mean(df['pred_conf'])
        # accuracy metrics
        if settings['task_name'] in ALL_CLS_TASKS:
            y_true = df['label']
            y_pred = df['pred_value']

            item['accuracy-acc'] = metrics.accuracy_score(y_true, y_pred)
            item['accuracy-f1'] = metrics.f1_score(y_true, y_pred)
            item['accuracy-precision'] = metrics.precision_score(y_true, y_pred)
            item['accuracy-recall'] = metrics.recall_score(y_true, y_pred)
            try:
                item['accuracy-auc'] = metrics.roc_auc_score(y_true, y_pred)
            except ValueError:
                item['accuracy-auc'] = np.nan

            y_bsl = bsl_df['pred_value']
            item['similarity-acc'] = metrics.accuracy_score(y_bsl, y_pred)
            item['similarity-f1'] = metrics.f1_score(y_bsl, y_pred)
            item['similarity-precision'] = metrics.precision_score(y_bsl, y_pred)
            item['similarity-recall'] = metrics.recall_score(y_bsl, y_pred)
            try:
                item['similarity-auc'] = metrics.roc_auc_score(y_bsl, y_pred)
            except ValueError:
                item['similarity-auc'] = np.nan

            item['accuracy'] = item['accuracy-acc']
            item['similarity'] = item['similarity-acc']
        elif settings['task_name'] in ALL_REG_TASKS:
            y_true = df['label']
            y_pred = df['pred_value']

            y_min, y_max = y_true.min(), y_true.max()
            y_pred = np.clip(y_pred, y_min, y_max)

            item['accuracy-mse'] = metrics.mean_squared_error(y_true, y_pred)
            item['accuracy-mae'] = metrics.mean_absolute_error(y_true, y_pred)
            item['accuracy-r2'] = metrics.r2_score(y_true, y_pred)
            item['accuracy-mape'] = metrics.mean_absolute_percentage_error(y_true, y_pred)
            item['accuracy-maxe'] = metrics.max_error(y_true, y_pred)

            y_bsl = bsl_df['pred_value']
            item['similarity-mse'] = metrics.mean_squared_error(y_bsl, y_pred)
            item['similarity-mae'] = metrics.mean_absolute_error(y_bsl, y_pred)
            item['similarity-r2'] = metrics.r2_score(y_bsl, y_pred)
            item['similarity-mape'] = metrics.mean_absolute_percentage_error(y_bsl, y_pred)
            item['similarity-maxe'] = metrics.max_error(y_bsl, y_pred)

            if item['system'] == 'exact':
                item['meet_rate'] = 1.0
            elif item['system'] == 'biathlon':
                assert item['condition_type'] == 'conf', f"condition_type = {item['condition_type']}"
                residule = (y_bsl - y_pred)
                max_error = float(item['max_error'])
                item['meet_rate'] = np.sum(np.abs(residule) <= max_error) / len(residule)

                if max_errors.get(item['task_name'], None) is None:
                    max_errors[item['task_name']] = []
                max_errors[item['task_name']].append(max_error)
            elif item['system'] == 'ralf':
                assert len(max_errors[item['task_name']]) > 0
                residule = (y_bsl - y_pred)
                for max_error in max_errors[item['task_name']]:
                    item[f'meet_rate_{max_error}'] = np.sum(np.abs(residule) <= max_error) / len(residule)
            else:
                raise ValueError(f"Unknown system: {item['system']}")

            item['accuracy'] = item['accuracy-mse']
            item['similarity'] = item['similarity-mse']
        else:
            raise ValueError(f"Unknown task: {settings['task_name']}")
        data.append(item)

    df = pd.DataFrame(data)
    df.to_csv(f"{args.out_dir}/full_{args.filename}", index=False)

    df = aggregate_data(args, df)
    df.to_csv(f"{args.out_dir}/avg_{args.filename}", index=False)
    print(f"stats saved to avg_{args.filename}")
