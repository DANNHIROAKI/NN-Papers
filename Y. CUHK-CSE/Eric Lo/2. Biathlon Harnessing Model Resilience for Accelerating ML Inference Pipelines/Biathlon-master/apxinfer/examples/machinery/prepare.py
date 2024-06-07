import pandas as pd
import os

from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker


class MachineryMultiClassPrepareWorker(XIPPrepareWorker):
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
        self.table = f"mach_imbalance_{nparts}"

    def get_requests(self) -> pd.DataFrame:
        sql = f"""
            SELECT bid
            FROM {self.database}.{self.table}
            GROUP BY bid
            ORDER BY bid
        """
        requests: pd.DataFrame = self.db_client.query_df(sql)
        if self.max_requests > 0 and self.max_requests < len(requests):
            requests = requests.sample(self.max_requests, replace=False)
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")
        sql = f"""
            SELECT bid as req_bid, label
            FROM (
                SELECT bid, label
                FROM {self.database}.{self.table}
                GROUP BY (bid, label)
                ORDER BY (bid, label)
                )
            WHERE bid IN ({','.join([str(x) for x in requests['req_bid'].values])})
        """
        df: pd.DataFrame = self.db_client.query_df(sql)
        labels = requests.merge(df, on="req_bid", how="left")
        labels.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels["label"]


class MachineryBinaryClassPrepareWorker(MachineryMultiClassPrepareWorker):
    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        labels = super().get_labels(requests)
        labels = labels.apply(lambda x: 1 if x > 0 else 0)
        return labels


class MachineryRalfPrepareWorker(MachineryBinaryClassPrepareWorker):
    def _extract_requests(self, max_num: int = 0) -> pd.DataFrame:
        sql = f"""
            SELECT bid as ts,
                bid as label_ts,
                bid as bid
            FROM {self.database}.{self.table}
            GROUP BY bid
            ORDER BY bid
        """
        requests: pd.DataFrame = self.db_client.query_df(sql)

        if max_num > 0 and max_num < len(requests):
            requests = requests.sample(max_num, replace=False)

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests()

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests


class MachineryRalfTestPrepareWorker(MachineryRalfPrepareWorker):
    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests(max_num=200)

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests


class MachineryRalfMedianPrepareWorker(MachineryRalfPrepareWorker):
    def create_dataset(self) -> pd.DataFrame:
        self.logger.info(f"Creating dataset for {self.dataset_dir} by copying")
        working_dir = self.working_dir
        elements = working_dir.split("/")
        task_name = None
        for ele in elements:
            if ele.startswith("machineryralf"):
                task_name = ele
                break
        assert task_name is not None
        machineryralf_dir = working_dir.replace(task_name, "machineryralf")
        assert os.path.exists(machineryralf_dir)
        machineryralf_dataset = pd.read_csv(
            os.path.join(machineryralf_dir, "dataset", "dataset.csv")
        )

        status = os.system(f"cp -r {machineryralf_dir}/model {working_dir}/")
        if status != 0:
            status = os.system(f"sudo cp -r {machineryralf_dir}/model {working_dir}/")
            if status != 0:
                raise ValueError("Failed to copy model directory")

        status = os.system(f"cp -r {machineryralf_dir}/qcosts.json {working_dir}/")
        if status != 0:
            status = os.system(
                f"sudo cp -r {machineryralf_dir}/qcosts.json {working_dir}/"
            )
            if status != 0:
                raise ValueError("Failed to copy qcosts.json")

        status = os.system(f"cp -r {machineryralf_dir}/../offline {working_dir}/../")
        if status != 0:
            status = os.system(
                f"sudo cp -r {machineryralf_dir}/../offline {working_dir}/../"
            )
            if status != 0:
                raise ValueError("Failed to copy offline directory")

        e2emedian_dir = working_dir.replace(
            f"/{task_name}/", "/machineryralfe2emedian01234567/"
        )
        assert os.path.exists(e2emedian_dir)
        if task_name.startswith("machineryralfdirectmedian"):
            fids = [int(v) for v in task_name[len("machineryralfdirectmedian") :]]
        elif task_name.startswith("machineryralfsimmedian"):
            fids = [int(v) for v in task_name[len("machineryralfsimmedian") :]]
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        if len(fids) == 0:
            fids = list(range(8))

        e2emedian_dataset = pd.read_csv(
            os.path.join(e2emedian_dir, "dataset", "dataset.csv")
        )

        median_fnames = [
            col
            for col in e2emedian_dataset.columns
            if col.startswith("f_") and "_median_" in col
        ]
        corres_avg_fname = [
            fname.replace("_median_", "_avg_") for fname in median_fnames
        ]
        assert len(corres_avg_fname) == len(median_fnames)
        assert len(corres_avg_fname) == 8

        dataset = machineryralf_dataset
        for i, fid in enumerate(fids):
            median_fname = median_fnames[i]
            avg_fname = corres_avg_fname[i]
            dataset = dataset.rename(columns={avg_fname: median_fname})
            req_col = f"req_offset_{median_fname[2:]}"
            dataset[req_col] = dataset[median_fname] - e2emedian_dataset[median_fname]

        self.logger.info(f"Created dataset for {self.dataset_dir}")
        return dataset
