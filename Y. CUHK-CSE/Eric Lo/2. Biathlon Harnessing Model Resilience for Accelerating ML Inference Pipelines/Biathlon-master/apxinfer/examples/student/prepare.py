import numpy as np
import pandas as pd
import os
import os.path as osp
import time
import json
from tqdm import tqdm
from typing import Tuple


from apxinfer.core.utils import XIPQType
from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker

from apxinfer.examples.student.data import get_dsrc
from apxinfer.examples.student.query import get_query_group
from apxinfer.examples.student.engine import STUDENT_CATEGORICAL, STUDENT_NUMERICAL


def feature_engineer(data_df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for c in STUDENT_CATEGORICAL:
        tmp = data_df.groupby(["session_id", "level_group"])[c].agg("nunique")
        tmp.name = tmp.name + "_unique"
        dfs.append(tmp)
    for c in STUDENT_NUMERICAL:
        tmp = data_df.groupby(["session_id", "level_group"])[c].agg("mean")
        tmp.name = tmp.name + "_avg"
        dfs.append(tmp)
    for c in STUDENT_NUMERICAL:
        tmp = data_df.groupby(["session_id", "level_group"])[c].agg("std")
        tmp.name = tmp.name + "_stdSamp"
        dfs.append(tmp)
    df = pd.concat(dfs, axis=1)
    df = df.fillna(-1)
    df = df.reset_index()
    # df = df.set_index("session_id")
    return df


class StudentPrepareWorker(XIPPrepareWorker):
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
        nparts: int,
    ) -> None:
        super().__init__(
            working_dir,
            fextractor,
            max_requests,
            train_ratio,
            valid_ratio,
            model_type,
            model_name,
            seed,
        )
        self.database = f"xip_{seed}"
        self.table = f"student_{nparts}"

        self._cached_df = None

    def extract_request_and_labels(self) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df
        dsrc = get_dsrc()
        data_df = pd.read_csv(os.path.join(dsrc, "train.csv"))
        df_features = feature_engineer(data_df)

        labels = pd.read_csv(os.path.join(dsrc, "train_labels.csv"))
        labels["qno"] = labels["session_id"].apply(lambda x: int(x.split("_")[-1][1:]))
        labels["session_id"] = labels["session_id"].apply(
            lambda x: int(x.split("_")[0])
        )
        df = labels
        df["level_group"] = df["qno"].apply(get_query_group)
        df = df.merge(df_features, on=["session_id", "level_group"], how="left")
        df = df.reset_index()

        df.insert(0, "ts", range(len(df)))
        df["label_ts"] = df["ts"]

        df.to_csv(os.path.join(self.working_dir, "extracted_df.csv"), index=False)
        self._cached_df = df
        return self._cached_df

    def get_requests(self) -> pd.DataFrame:
        df = self.extract_request_and_labels()
        requests = df[["ts", "label_ts", "session_id", "qno"]]

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_features(self, requests: pd.DataFrame) -> pd.DataFrame:
        nreqs = len(requests)
        df = self.extract_request_and_labels()
        self.logger.info(f"Getting features for {nreqs}x requests")
        df = requests.merge(
            df,
            left_on=["req_session_id", "req_qno"],
            right_on=["session_id", "qno"],
            how="left",
        )
        df["req_qno_0"] = df["req_qno"]
        fnames = []
        qfeatures_list = []
        qcosts = []
        for qid, query in tqdm(enumerate(self.fextractor.queries)):
            st = time.time()
            fnames.extend(query.fnames)

            self.logger.info(f"Extracting features {query.fnames}")
            # qtype = query.qtype
            # if qtype == XIPQType.AGG:
            #     cols = [
            #         "_".join(fname.split("_")[1:-1]) for fname in query.fnames
            #     ]
            # elif qtype == XIPQType.NORMAL:
            #     cols = [
            #         "_".join(fname.split("_")[1:-1]) for fname in query.fnames
            #     ]
            # else:
            #     raise ValueError(f"Unknown query type {qtype}")
            cols = ["_".join(fname.split("_")[1:-1]) for fname in query.fnames]
            qfeatures = df[cols].values
            self.logger.info(f"Extracted features {query.fnames}")

            qfeatures_list.append(qfeatures)
            qcosts.append(time.time() - st)
        features = np.concatenate(qfeatures_list, axis=1)
        features = pd.DataFrame(features, columns=fnames)
        with open(osp.join(self.working_dir, "qcosts.json"), "w") as f:
            json.dump({"nreqs": nreqs, "qcosts": qcosts}, f, indent=4)
        features.to_csv(osp.join(self.working_dir, "features.csv"), index=False)
        return features

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        df = self.extract_request_and_labels()
        df = requests.merge(
            df,
            left_on=["req_session_id", "req_qno"],
            right_on=["session_id", "qno"],
            how="left",
        )
        labels = df[["correct"]]
        self.logger.info(f"Getting labels for {len(labels)}x requests")
        labels.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels["correct"]

    def get_test_users(self, USER_LIST, split):
        rng = np.random.RandomState(0)
        users_test = USER_LIST[split:][
            rng.choice(
                len(USER_LIST[split:]), len(USER_LIST[split:]) // 200, replace=False
            )
        ]
        return users_test

    def split_dataset(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dataset.set_index("req_session_id", inplace=True)
        USER_LIST = dataset.index.unique()
        split = int(len(USER_LIST) * (1 - 0.20))
        train_set, valid_set = (
            dataset.loc[USER_LIST[:split]],
            dataset.loc[USER_LIST[split:]],
        )
        users_test = self.get_test_users(USER_LIST, split)
        test_set = dataset.loc[users_test]
        # reset index
        train_set.reset_index(inplace=True)
        valid_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)
        return train_set, valid_set, test_set


class StudentQNoPrepareWorker(StudentPrepareWorker):
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
        nparts: int,
        qno: int,
    ) -> None:
        self.qno = qno
        super().__init__(
            working_dir,
            fextractor,
            max_requests,
            train_ratio,
            valid_ratio,
            model_type,
            model_name,
            seed,
            nparts,
        )

    def get_requests(self) -> pd.DataFrame:
        df = self.extract_request_and_labels()
        df = df[df["qno"] == self.qno]
        requests = df[["ts", "label_ts", "session_id", "qno"]]
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_test_users(self, USER_LIST, split):
        rng = np.random.RandomState(0)
        users_test = USER_LIST[split:][
            rng.choice(
                len(USER_LIST[split:]), len(USER_LIST[split:]) // 10, replace=False
            )
        ]
        return users_test


def feature_engineer_median(data_df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for c in STUDENT_CATEGORICAL:
        tmp = data_df.groupby(["session_id", "level_group"])[c].agg("nunique")
        tmp.name = tmp.name + "_unique"
        dfs.append(tmp)
    for c in STUDENT_NUMERICAL:
        tmp = data_df.groupby(["session_id", "level_group"])[c].agg("median")
        tmp.name = tmp.name + "_median"
        dfs.append(tmp)
    for c in STUDENT_NUMERICAL:
        tmp = data_df.groupby(["session_id", "level_group"])[c].agg("std")
        tmp.name = tmp.name + "_stdSamp"
        dfs.append(tmp)
    df = pd.concat(dfs, axis=1)
    df = df.fillna(-1)
    df = df.reset_index()
    # df = df.set_index("session_id")
    return df


class StudentQNoMedianPrepareWorker(StudentQNoPrepareWorker):
    def extract_request_and_labels(self) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df
        dsrc = get_dsrc()
        data_df = pd.read_csv(os.path.join(dsrc, "train.csv"))
        df_features = feature_engineer_median(data_df)

        labels = pd.read_csv(os.path.join(dsrc, "train_labels.csv"))
        labels["qno"] = labels["session_id"].apply(lambda x: int(x.split("_")[-1][1:]))
        labels["session_id"] = labels["session_id"].apply(
            lambda x: int(x.split("_")[0])
        )
        df = labels
        df["level_group"] = df["qno"].apply(get_query_group)
        df = df.merge(df_features, on=["session_id", "level_group"], how="left")
        df = df.reset_index()

        df.insert(0, "ts", range(len(df)))
        df["label_ts"] = df["ts"]

        df.to_csv(os.path.join(self.working_dir, "extracted_df.csv"), index=False)
        self._cached_df = df
        return self._cached_df


class StudentQNoSimMedianPrepareWorker(StudentQNoMedianPrepareWorker):
    def create_dataset(self) -> pd.DataFrame:
        return self.create_dataset_simmedian_helper(ref_task="studentqnov2subset")


class StudentQNoV2PrepareWorker(StudentQNoPrepareWorker):
    def get_test_users(self, USER_LIST, split):
        return USER_LIST[split:]


class StudentQNoTestPrepareWorker(StudentQNoPrepareWorker):
    def get_test_users(self, USER_LIST, split):
        users = super().get_test_users(USER_LIST, split)
        return users[:50]
