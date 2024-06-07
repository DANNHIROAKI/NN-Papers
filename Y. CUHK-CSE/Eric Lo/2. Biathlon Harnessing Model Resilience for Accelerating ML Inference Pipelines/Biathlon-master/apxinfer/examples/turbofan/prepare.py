import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker

from apxinfer.examples.turbofan.data import get_dsrc


class TurbofanPrepareWorker(XIPPrepareWorker):
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
        self.table = f"turbofan_{nparts}"

        self._cached_df = None

    def extract_request_and_labels(self) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df
        dsrc = get_dsrc()
        train_stats_folder = os.path.join(dsrc, "train_stats", "train_stats")
        dfs = []
        for filename in tqdm(os.listdir(train_stats_folder)):
            df = pd.read_csv(os.path.join(train_stats_folder, filename))
            df["name"] = filename.split(".")[0].replace("_stats", "")
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        # unit-cycle => unit, cycle.
        # unit <= unit-cycle.split('-')[0]
        # cycle <= unit-cycle.split('-')[1]
        df["unit"] = df["unit-cycle"].apply(lambda x: int(x.split("-")[0]))
        df["cycle"] = df["unit-cycle"].apply(lambda x: int(x.split("-")[1]))
        df = df.dropna()
        self._cached_df = df
        df.to_csv(os.path.join(self.working_dir, "extracted_df.csv"), index=False)
        return self._cached_df

    def get_requests(self) -> pd.DataFrame:
        df = self.extract_request_and_labels()
        requests = df[["name", "unit", "cycle"]]
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        df = self.extract_request_and_labels()
        labels = df[["Y"]]
        self.logger.info(f"Getting labels for {len(labels)}x requests")
        labels.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels["Y"]


class TurbofanSimMedianPrepareWorker(TurbofanPrepareWorker):
    def create_dataset(self) -> pd.DataFrame:
        return self.create_dataset_simmedian_helper(ref_task="turbofan")
