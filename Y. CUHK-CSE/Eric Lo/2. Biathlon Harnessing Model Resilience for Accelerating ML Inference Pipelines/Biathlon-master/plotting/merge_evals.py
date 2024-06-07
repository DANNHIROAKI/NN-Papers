import os
import sys
import json
import pandas as pd
from tap import Tap
from tqdm import tqdm
import numpy as np

# read all csv files in the directory, and merge them into one dataframe
# parse the filename to get the task name, model name, nparts, ncfgs, ncores, and max_error,
# and add them as columns to the dataframe
# save the dataframe to a csv file


class EvalArgs(Tap):
    csv_dir: str = "./cache/evals"
    output_dir: str = "./cache"
    output_filename: str = "evals.csv"
    recursive: bool = False
    avg: bool = False


select_names = ["Trips-Fare",
                # "battery",
                "batteryv2",
                "turbofan",
                "Bearing-MLP",
                # "Bearing-Multi",
                "Fraud-Detection",
                # "tdfraudrandom",
                'student'
                ]
select_names += [f'machineryxf{i}' for i in range(1, 9)]
select_names += [f'machinerynf{i}' for i in range(1, 9)]
select_names += [f'studentqno{i}' for i in range(1, 19)]
select_names += [f'studentqno18nf{i}' for i in range(1, 13)]
# select_names += [f'tickvaryNM{i}' for i in range(2, 40)]


def parse_filename(filename, verbose: bool = False):
    if verbose:
        print(filename)
    # remove the extension
    filename = filename.replace(".csv", "")
    elements = filename.split("_")
    task_name = elements[0]
    _ = elements[1]
    model_name = elements[2]
    nparts = int(elements[3])
    ncfgs = int(elements[4])
    ncores = int(elements[5])
    max_error = float(elements[6])
    if task_name == "tick":
        task_name = "tick-v1"
    elif task_name == "tickv2":
        task_name = "tick-v2"
    elif task_name == "tripsfeast":
        task_name = "Trips-Fare"
    elif task_name == "machinery":
        if model_name == "mlp":
            task_name = "Bearing-MLP"
        elif model_name == "dt":
            task_name = "machinery-v2"
        elif model_name == "knn":
            task_name = "Bearing-KNN"
    elif task_name == "machinerymulti":
        task_name = "Bearing-Multi"
    elif task_name == "tdfraud":
        task_name = "Fraud-Detection"
    return task_name, model_name, nparts, ncfgs, ncores, max_error


def merge_csv(args: EvalArgs):
    csv_dir = args.csv_dir
    # get all files in the directory
    files = []
    if args.recursive:
        for root, dirs, filenames in os.walk(csv_dir):
            for filename in filenames:
                if filename.endswith(".csv"):
                    files.append(os.path.join(root, filename))
    else:
        files = [os.path.join(csv_dir, filename) for filename in os.listdir(csv_dir) if filename.endswith(".csv")]
    print(f"Found {len(files)} csv files in {csv_dir}")

    df = pd.DataFrame()
    for fpath in tqdm(files):
        if fpath.endswith(".csv"):
            fname = os.path.basename(fpath)
            task_name, model_name, nparts, ncfgs, ncores, max_error = parse_filename(fname)
            if task_name not in select_names:
                continue
            df_tmp = pd.read_csv(os.path.join(fpath))
            df_tmp["task_name"] = task_name
            df_tmp["model_name"] = model_name
            # df_tmp["nparts"] = nparts # already in the csv file
            df_tmp["ncfgs"] = ncfgs
            # df_tmp["ncores"] = ncores # already in the csv file
            # df_tmp["max_error"] = max_error # already in the csv file
            # move the columns to the front
            cols = df_tmp.columns.tolist()
            cols = cols[-3:] + cols[:-3]
            df_tmp = df_tmp[cols]

            if "pest_nsamples" not in df_tmp.columns:
                df_tmp["pest_nsamples"] = 1000

            # set accuracy
            if task_name in ["tick-v1", "tick-v2", "tickvaryNM1",
                             "tickvaryNM8", "turbofan",
                             "battery", "batteryv2",
                             "Tick-Price", "tripsfeast", "Trips-Fare"]:
                acc_type = "r2"
                if f"accuracy-{acc_type}" in df_tmp.columns:
                    df_tmp['accuracy'] = df_tmp[f"accuracy-{acc_type}"]
                    df_tmp['similarity'] = df_tmp[f"similarity-{acc_type}"]
            if task_name in ['cheaptrips', "machinery-v1", "machinery-v2",
                             "machinery-v3", "machinerymulti", "tdfraud",
                             "tdfraudrandom",
                             "Fraud-Detection", "Bearing-MLP",
                             "Bearing-Multi", "Bearing-KNN",]:
                acc_type = "f1"
                if f"accuracy-{acc_type}" in df_tmp.columns:
                    df_tmp['accuracy'] = df_tmp[f"accuracy-{acc_type}"]
                    df_tmp['similarity'] = df_tmp[f"similarity-{acc_type}"]

            # get exact acc from the row with 1.0 min_conf
            exact_acc = df_tmp[df_tmp['min_conf'] == 1.0]['accuracy'].values[0]
            df_tmp['acc_loss'] = exact_acc - df_tmp['accuracy']
            df_tmp['acc_loss_pct'] = df_tmp['acc_loss'] / exact_acc
            df_tmp['acc_diff'] = abs(df_tmp['acc_loss'])
            # list_cols : ["avg_sample_query", "avg_qtime_query"]
            df_tmp['avg_sample_query'] = df_tmp['avg_sample_query'].apply(lambda x: json.loads(x))
            df_tmp['avg_qtime_query'] = df_tmp['avg_qtime_query'].apply(lambda x: json.loads(x))
            df = pd.concat([df, df_tmp])
    return df


def seed_selection(df: pd.DataFrame) -> pd.DataFrame:
    seeds_dict = {
        "Trips-Fare": [1, 2, 3],
        # "tickvaryNM8": [0, 1, 2],
        # "battery": [0, 4, 3],
        "batteryv2": [3, 4, 1],
        "turbofan": [1, 2, 3],
        "Bearing-MLP": [0, 2, 4],
        "Fraud-Detection": [0, 1, 2],
        # "tdfraudrandom": [1, 3, 4],
        "tdfraudrandom": [0],
    }
    df = df[(df['task_name'] != 'Trips-Fare') | ((df['task_name'] == 'Trips-Fare') & df['seed'].isin(seeds_dict['Trips-Fare']))]
    # df = df[(df['task_name'] != 'tickvaryNM8') | ((df['task_name'] == 'tickvaryNM8') & df['seed'].isin(seeds_dict['tickvaryNM8']))]
    # df = df[(df['task_name'] != 'battery') | ((df['task_name'] == 'battery') & df['seed'].isin(seeds_dict['battery']))]
    df = df[(df['task_name'] != 'batteryv2') | ((df['task_name'] == 'batteryv2') & df['seed'].isin(seeds_dict['batteryv2']))]
    df = df[(df['task_name'] != 'turbofan') | ((df['task_name'] == 'turbofan') & df['seed'].isin(seeds_dict['turbofan']))]
    df = df[(df['task_name'] != 'Bearing-MLP') | ((df['task_name'] == 'Bearing-MLP') & df['seed'].isin(seeds_dict['Bearing-MLP']))]
    df = df[(df['task_name'] != 'Fraud-Detection') | ((df['task_name'] == 'Fraud-Detection') & df['seed'].isin(seeds_dict['Fraud-Detection']))]
    df = df[(df['task_name'] != 'tdfraudrandom') | ((df['task_name'] == 'tdfraudrandom') & df['seed'].isin(seeds_dict['tdfraudrandom']))]
    return df


def tmp_handler_for_varynm(df: pd.DataFrame) -> pd.DataFrame:
    # for task that starts with tickvaryNM, set the baseline AFC as 0.5385772151540417
    # baseline is those rows with min_conf = 1.0
    varyNM_tasks = [f"tickvaryNM{i}" for i in range(1, 40)]
    for task_name in varyNM_tasks:
        # for those with min_conf=1.0, set BD:AFC as 0.5385772151540417, and update avg_latency accordingly
        # for those with min_conf<1.0, update speedup accordingly
        df_tmp = df[df["task_name"] == task_name].copy()
        if len(df_tmp) == 0:
            continue

        df_tmp_baseline = df_tmp[df_tmp["min_conf"] == 1.0].copy()
        old_afc = df_tmp_baseline["BD:AFC"].values[0]
        df_tmp_baseline["BD:AFC"] = 0.5385772151540417
        df_tmp_baseline["avg_latency"] += (0.5385772151540417 - old_afc)

        baseline_lat = df_tmp_baseline["avg_latency"].values[0]
        df_tmp_baseline["speedup"] = df_tmp_baseline["avg_latency"] / baseline_lat

        df_tmp_others = df_tmp[df_tmp["min_conf"] != 1.0].copy()
        df_tmp_others["speedup"] = baseline_lat / df_tmp_others["avg_latency"]

        df = df[df["task_name"] != task_name]
        df = pd.concat([df, df_tmp_baseline, df_tmp_others])
    return df


def main():
    args = EvalArgs().parse_args()
    raw_df = merge_csv(args)
    print(f'columns: {raw_df.columns}')
    useless_cols = ['run_shared', 'nocache', 'interpreter', 'min_confs']
    df = raw_df.drop(columns=useless_cols, errors='ignore')
    df = seed_selection(df)
    # df = tmp_handler_for_varynm(df)
    if args.avg:
        # seed,
        # agg_qids,task_home,
        # sampling_rate,avg_latency,speedup, BD:AFC,BD:AMI,BD:Sobol,
        # similarity-maxe,accuracy-maxe, similarity-r2,accuracy-r2,similarity-mse,accuracy-mse,similarity-mape,accuracy-mape,
        # similarity-auc,accuracy-auc,similarity-acc,accuracy-acc,similarity-f1,accuracy-f1
        # similarity, accuracy, acc_loss,acc_loss_pct,acc_diff,
        keys = ['task_name', 'model_name', 'model',
                'task_home', 'agg_qids',
                'qinf', 'policy', 'nparts', 'ncfgs',
                'ncores', 'loading_mode',
                'scheduler_batch', 'scheduler_init',
                'max_error', 'min_conf', 'pest_nsamples']
        # deduplicate by the keys + ['seed'], keep the first
        print(f'total number of rows: {len(df)}')
        df = df.groupby(keys + ['seed']).first().reset_index()
        print(f'total number of rows: {len(df)}')
        df_grp = df.groupby(keys)

        verbose = True
        if verbose:
            # select the rows with task_name=Fraud-Detection, 
            # loading_mode=0, scheduler_batch=3, scheduler_init=1, 
            # min_conf in [0.98, 0.99, 0.995, 0.999]
            tmp_rows = df[df['task_name'] == 'Fraud-Detection']
            tmp_rows = tmp_rows[tmp_rows['loading_mode'] == 0]
            tmp_rows = tmp_rows[tmp_rows['scheduler_init'] == 1]
            tmp_rows = tmp_rows[tmp_rows['scheduler_batch'] == 3]
            # tmp_rows = tmp_rows[tmp_rows['min_conf'].isin([0.95, 0.98, 0.99, 0.995, 0.999])]
            # tmp_rows = tmp_rows[tmp_rows['min_conf'].isin([0.6, 0.7, 0.8, 0.9, 0.95])]
            tmp_rows = tmp_rows[tmp_rows['min_conf'].isin([0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])]
            tmp_rows = tmp_rows.sort_values(by=['min_conf', 'seed'])
            print(tmp_rows[["min_conf", 'seed', 'avg_latency', "speedup", "similarity"]])

        list_cols = ["avg_sample_query", "avg_qtime_query"]
        non_list_cols = [col for col in df.columns if col not in list_cols + keys]
        # for columns except list_cols, take the mean;
        # for list_cols, take the mean of each element
        df_non_list = df_grp[non_list_cols].mean().reset_index()
        for col in list_cols:
            df_list = df_grp[col].apply(lambda x: np.mean(x.tolist(), axis=0).tolist()).reset_index()
        df = pd.merge(df_non_list, df_list, on=keys)

    # add column naggs, which = len(agg_qids)
    df['naggs'] = df['agg_qids'].apply(lambda x: len(json.loads(x)))
    print(f'total number of rows: {len(df)}')
    df.to_csv(os.path.join(args.output_dir, args.output_filename), index=False)


if __name__ == "__main__":
    main()
