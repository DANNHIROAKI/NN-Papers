import numpy as np
import pandas as pd
import time
import os
import os.path as osp
import logging
import json
from tqdm import tqdm
from typing import Tuple

from apxinfer.core.utils import XIPFeatureVec
from apxinfer.core.data import DBHelper
from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor

logging.basicConfig(level=logging.INFO)


def train_valid_test_split(
    dataset: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # shuffle the data
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    # calculate the number of rows for each split
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    # n_test = n_total - n_train - n_valid

    # split the data
    train_set = dataset[:n_train]
    valid_set = dataset[n_train : n_train + n_valid]
    test_set = dataset[n_train + n_valid :]

    return train_set, valid_set, test_set


class XIPPrepareWorker:
    """This Worker prepares dataset for model training and evaluation
    It will prepare requests, labels, queries, and features
    for training, validation and testing
    """

    def __init__(
        self,
        working_dir: str,
        fextractor: XIPFeatureExtractor,
        max_requests: int,
        train_ratio: float,
        valid_ratio: float,
        model_type: str,
        model_name: str,
        seed: int,
        split_seed: int = 0,
    ) -> None:
        self.working_dir = working_dir

        self.fextractor = fextractor
        self.db_client = DBHelper.get_db_client()

        self.max_requests = max_requests
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        self.model_type = model_type
        self.model_name = model_name

        self.seed = seed  # to identify db
        self.split_seed = split_seed  # for dataset split
        self.logger = logging.getLogger("DatasetCreator")

        self.dataset_dir = osp.join(self.working_dir, "dataset")
        os.makedirs(self.dataset_dir, exist_ok=True)

    def get_requests(self) -> pd.DataFrame:
        self.logger.info(f"Getting requests for {osp.basename(self.working_dir)}")
        raise NotImplementedError

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")
        raise NotImplementedError

    def get_features(self, requests: pd.DataFrame) -> pd.DataFrame:
        nreqs = len(requests)
        self.logger.info(f"Getting features for {nreqs}x requests")
        fnames = []
        qfeatures_list = []
        qcosts = []
        for qid, query in enumerate(self.fextractor.queries):
            st = time.time()
            num_qf = len(query.fnames)
            fnames.extend(query.fnames)

            qfeatures = np.zeros((nreqs, num_qf))
            self.logger.info(f"Extracting features {query.fnames}")
            final_qcfg = query.get_qcfg(100, 1.0)
            for rid, req in tqdm(
                enumerate(requests.to_dict(orient="records")),
                desc=f"Extracting {qid}",
                total=nreqs,
            ):
                fvec: XIPFeatureVec = query.run(req, final_qcfg)
                qfeatures[rid] = fvec["fvals"]
            self.logger.info(f"Extracted features {query.fnames}")
            qfeatures_list.append(qfeatures)
            qcosts.append(time.time() - st)
        features = np.concatenate(qfeatures_list, axis=1)
        features = pd.DataFrame(features, columns=fnames)
        with open(osp.join(self.working_dir, "qcosts.json"), "w") as f:
            json.dump({"nreqs": nreqs, "qcosts": qcosts}, f, indent=4)
        features.to_csv(osp.join(self.working_dir, "features.csv"), index=False)
        return features

    def create_dataset(self) -> pd.DataFrame:
        # return dataset, fnames, label_name
        self.logger.info(f"Creating dataset for {self.dataset_dir}")
        requests = self.get_requests()
        requests = requests.add_prefix("req_")
        # add request_id column
        requests.insert(0, "req_id", range(len(requests)))

        features = self.get_features(requests)
        features = features.add_prefix("f_")

        labels = self.get_labels(requests)
        labels = labels.rename("label")

        dataset = pd.concat(
            [
                requests.reset_index(drop=True),
                features.reset_index(drop=True),
                labels.reset_index(drop=True),
            ],
            axis=1,
        )

        # remove the requests that have no features or labels
        dataset = dataset.dropna()
        self.logger.info(f"droped {len(dataset) - len(requests)}x requests")

        return dataset

    def split_dataset(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_set, valid_set, test_set = train_valid_test_split(
            dataset=dataset,
            train_ratio=self.train_ratio,
            valid_ratio=self.valid_ratio,
            seed=self.split_seed,
        )
        return train_set, valid_set, test_set

    def run(self, skip_dataset: bool = False) -> None:
        if skip_dataset and osp.exists(osp.join(self.dataset_dir, "dataset.csv")):
            dataset = pd.read_csv(osp.join(self.dataset_dir, "dataset.csv"))
        else:
            dataset = self.create_dataset()
            dataset.to_csv(osp.join(self.dataset_dir, "dataset.csv"), index=False)

        train_set, valid_set, test_set = self.split_dataset(dataset)
        # save the dataset
        self.logger.info(f"Saving dataset for {self.dataset_dir}")
        train_set.to_csv(osp.join(self.dataset_dir, "train_set.csv"), index=False)
        valid_set.to_csv(osp.join(self.dataset_dir, "valid_set.csv"), index=False)
        test_set.to_csv(osp.join(self.dataset_dir, "test_set.csv"), index=False)

        # save dataset statistics
        self.logger.info(f"Saving dataset statistics for {self.dataset_dir}")
        train_set.describe().to_csv(osp.join(self.dataset_dir, "train_set_stats.csv"))
        valid_set.describe().to_csv(osp.join(self.dataset_dir, "valid_set_stats.csv"))
        test_set.describe().to_csv(osp.join(self.dataset_dir, "test_set_stats.csv"))

        rec_budget = self.recommend_ralf_budget(test_set)
        self.logger.info(f"Recommended RALF budget: {rec_budget}")

    def recommend_ralf_budget(self, dataset: pd.DataFrame) -> float:
        if "req_ts" not in dataset.columns:
            return 0.0
        # compute the mean time interval on req_ts
        interval = dataset["req_ts"].diff().mean()
        return 1.0 / interval

    def create_dataset_simmedian_helper(self, ref_task: str,
                                        replaced: str = 'avg') -> pd.DataFrame:
        self.logger.info(f"Creating dataset for {self.dataset_dir} by copying")
        working_dir = self.working_dir
        elements = working_dir.split("/")
        task_name = None
        for ele in elements:
            if ele.startswith(ref_task):
                task_name = ele
                break
        assert (
            task_name is not None
        ), f"Task name not found in {working_dir} with {ref_task}"

        base_ppl_dir = working_dir.replace(task_name, ref_task)
        assert os.path.exists(base_ppl_dir)
        base_ppl_dataset = pd.read_csv(
            os.path.join(base_ppl_dir, "dataset", "dataset.csv")
        )

        status = os.system(f"cp -r {base_ppl_dir}/model {working_dir}/")
        if status != 0:
            status = os.system(f"sudo cp -r {base_ppl_dir}/model {working_dir}/")
            if status != 0:
                raise ValueError("Failed to copy model directory")

        status = os.system(f"cp -r {base_ppl_dir}/qcosts.json {working_dir}/")
        if status != 0:
            status = os.system(f"sudo cp -r {base_ppl_dir}/qcosts.json {working_dir}/")
            if status != 0:
                raise ValueError("Failed to copy qcosts.json")

        status = os.system(f"cp -r {base_ppl_dir}/../offline {working_dir}/../")
        if status != 0:
            status = os.system(
                f"sudo cp -r {base_ppl_dir}/../offline {working_dir}/../"
            )
            if status != 0:
                raise ValueError("Failed to copy offline directory")

        e2emedian_dir = working_dir.replace(f"/{task_name}/", f"/{ref_task}median/")
        e2emedian_dataset = pd.read_csv(
            os.path.join(e2emedian_dir, "dataset", "dataset.csv")
        )

        median_fnames = [
            col
            for col in e2emedian_dataset.columns
            if col.startswith("f_") and "_median_" in col
        ]
        corres_avg_fname = [
            fname.replace("_median_", f"_{replaced}_") for fname in median_fnames
        ]

        assert len(base_ppl_dataset) == len(
            e2emedian_dataset
        ), f"Length of base_ppl_dataset({len(base_ppl_dataset)}) and e2emedian_dataset({len(e2emedian_dataset)}) is not same"

        dataset = base_ppl_dataset
        for i in range(len(median_fnames)):
            median_fname = median_fnames[i]
            avg_fname = corres_avg_fname[i]
            dataset = dataset.rename(columns={avg_fname: median_fname})
            req_col = f"req_offset_{median_fname[2:]}"
            dataset[req_col] = dataset[median_fname] - e2emedian_dataset[median_fname]

        self.logger.info(f"Created dataset for {self.dataset_dir}")
        return dataset
