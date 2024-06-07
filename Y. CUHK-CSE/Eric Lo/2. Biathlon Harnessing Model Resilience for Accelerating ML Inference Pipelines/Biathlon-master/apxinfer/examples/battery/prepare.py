import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker

from apxinfer.examples.battery.data import get_dsrc


class BatteryPrepareWorker(XIPPrepareWorker):
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
        self.table = f"battery_{nparts}"

        self._cached_df = None

    def extract_request_and_labels(self) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df
        dsrc = get_dsrc()
        meta_data = pd.read_csv(os.path.join(dsrc, "metadata.csv"))
        # selected_type = 'discharge'
        selected_type = "charge"
        selected_data = meta_data[meta_data["type"] == selected_type]

        rng = np.random.RandomState(0)
        selected_dfs = []
        for row in tqdm(selected_data.to_dict(orient="records")):
            filename = row["filename"]
            bid = row["uid"]
            src = os.path.join(dsrc, "data", filename)

            df = pd.read_csv(os.path.join(dsrc, "data", src))
            # pick a row randomly starting from the middle
            row = rng.randint(len(df) // 2, len(df))
            # get the time of that row
            time = df.iloc[row]["Time"]
            # get rul = last_row_time - time
            rul = df.iloc[-1]["Time"] - time
            features = {"bid": bid, "time": time, "rul": rul}

            aggops = ["min", "max", "mean", "std", "skew", "kurtosis"]
            aggops = ["mean", "std", "skew", "kurtosis"]
            aggops = ["mean", "std"]
            aggs = df.iloc[:row].agg(aggops).to_dict()
            for k, v in aggs.items():
                if k == "Time":
                    continue
                for kk, vv in v.items():
                    if kk == "mean":
                        features[f"{k}_avg"] = vv
                    elif kk == "std":
                        features[f"{k}_stdPop"] = vv
                    else:
                        features[f"{k}_{kk}"] = vv

            selected_dfs.append(pd.DataFrame([features]))

        df = pd.concat(selected_dfs)
        df = df.dropna()
        self._cached_df = df
        df.to_csv(os.path.join(self.working_dir, "extracted_df.csv"), index=False)
        return self._cached_df

    def get_requests(self) -> pd.DataFrame:
        df = self.extract_request_and_labels()
        requests = df[["bid", "time"]]
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        df = self.extract_request_and_labels()
        labels = df[["rul"]]
        self.logger.info(f"Getting labels for {len(labels)}x requests")
        labels.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels["rul"]


class BatteryTestPrepareWorker(BatteryPrepareWorker):
    pass


class BatterySimMedianPrepareWorker(BatteryPrepareWorker):
    def create_dataset(self) -> pd.DataFrame:
        return self.create_dataset_simmedian_helper(ref_task="batteryv2")
