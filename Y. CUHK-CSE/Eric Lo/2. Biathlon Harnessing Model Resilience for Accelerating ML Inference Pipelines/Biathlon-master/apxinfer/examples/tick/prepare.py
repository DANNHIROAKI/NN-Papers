import pandas as pd
import numpy as np
import datetime as dt
import os
from tqdm import tqdm
from typing import Tuple

from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker


class TickPrepareWorker(XIPPrepareWorker):
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
        self.table = f"tick_{nparts}"

    def get_requests(self) -> pd.DataFrame:
        self.logger.info("Getting requests")
        sql = f"""SELECT toString(min(tick_dt)) from {self.database}.{self.table}"""
        min_dt: str = self.db_client.command(sql)
        min_dt = dt.datetime.strptime(min_dt, "%Y-%m-%d %H:%M:%S.%f")
        start_dt = min_dt + dt.timedelta(hours=7)

        max_dt: str = self.db_client.command(
            f"""SELECT toString(max(tick_dt)) from {self.database}.{self.table}"""
        )
        max_dt = dt.datetime.strptime(max_dt, "%Y-%m-%d %H:%M:%S.%f")
        end_dt = max_dt - dt.timedelta(hours=1)

        sql = f""" WITH makeDateTime64(year, month, day, hour, 0, 0) as dt
                SELECT count()
                FROM xip.tick_fstore_hour
                WHERE dt > '{start_dt}' AND dt < '{end_dt}'
            """
        cnt: int = self.db_client.command(sql)
        self.logger.info(f"number of possible requests: {cnt}")
        rate = 1
        if self.max_requests > 0 and self.max_requests < cnt:
            rate = int(max(rate, cnt // self.max_requests))
        self.logger.info(f"requests sampling: {rate}")

        sql = f"""
                WITH makeDateTime64(year, month, day, hour, 0, 0) as clk_dt
                SELECT cpair, toString(clk_dt) as dt
                FROM xip.tick_fstore_hour
                WHERE (cityHash64(clk_dt) % {int(rate)}) == 0
                        AND clk_dt > '{start_dt}'
                        AND clk_dt < '{end_dt}'
                ORDER BY (cpair, year, month, day, hour)
                """
        df: pd.DataFrame = self.db_client.query_df(sql)
        self.logger.info(f"Got of {len(df)}x of requests")
        df.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return df

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")

        labels = []
        for req in tqdm(
            requests.to_dict(orient="records"),
            desc="Getting labels",
            total=len(requests),
        ):
            cpair: str = req["req_cpair"]
            req_dt: dt.datetime = pd.to_datetime(req["req_dt"])

            target_dt = req_dt + dt.timedelta(hours=1)
            year = target_dt.year
            month = target_dt.month
            day = target_dt.day
            hour = target_dt.hour
            avg_bid = self.db_client.command(
                f""" SELECT avg_bid
                    FROM xip.tick_fstore_hour
                    WHERE cpair = '{cpair}'
                    AND year = {year} AND month = {month}
                    AND day = {day} AND hour = {hour}
                """
            )
            labels.append(avg_bid)
        labels_pds = pd.Series(labels)
        labels_pds = pd.to_numeric(labels_pds, errors="coerce")
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds


class TickRalfPrepareWorker(TickPrepareWorker):
    def _extract_requests(
        self,
        start_dt: str = "2022-02-01 00:00:00.000",
        end_dt: str = "2022-03-01 00:00:00.000",
        sampling_rate: float = 1,
        max_num: int = 0,
    ) -> pd.DataFrame:
        self.logger.info("Getting requests")

        sql = f""" WITH makeDateTime64(year, month, day, hour, 0, 0) as dt
                SELECT count()
                FROM xip.tick_fstore_hour
                WHERE dt >= '{start_dt}' AND dt <= '{end_dt}'
            """
        cnt: int = self.db_client.command(sql)
        self.logger.info(f"number of possible requests: {cnt}")
        self.logger.info(f"requests sampling: {sampling_rate}")

        sql = f"""
                WITH makeDateTime64(year, month, day, hour, 0, 0) as clk_dt
                SELECT
                    toString(clk_dt) as ts,
                    toString(addHours(clk_dt, 1)) as label_ts,
                    cpair, toString(clk_dt) as dt
                FROM xip.tick_fstore_hour
                WHERE (cityHash64(clk_dt) % {int(1.0 / sampling_rate)}) == 0
                        AND clk_dt >= '{start_dt}'
                        AND clk_dt <= '{end_dt}'
                ORDER BY (cpair, year, month, day, hour)
                """
        requests: pd.DataFrame = self.db_client.query_df(sql)
        requests["ts"] = pd.to_datetime(requests["ts"]).astype(int) // 10**9
        requests["label_ts"] = pd.to_datetime(requests["label_ts"]).astype(int) // 10**9

        if max_num > 0 and max_num < len(requests):
            requests = requests[:max_num]

        self.logger.info(f"Got of {len(requests)}x of requests")
        return requests

    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests()

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def split_dataset(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n_train = len(dataset) // 2
        n_valid = 100
        train_set = dataset[:n_train]
        test_set = dataset[n_train:]
        valid_set = test_set[:n_valid]
        return train_set, valid_set, test_set


class TickRalfTestPrepareWorker(TickRalfPrepareWorker):
    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests(max_num=self.max_requests)

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests


class TickRalfV2PrepareWorker(TickRalfPrepareWorker):
    def _extract_requests(
        self,
        start_dt: str = "2022-02-01 00:00:00.000",
        end_dt: str = "2022-03-01 00:00:00.000",
        sampling_rate: float = 1,
        max_num: int = 0,
    ) -> pd.DataFrame:
        self.logger.info("Getting requests")

        sql = f""" WITH makeDateTime64(year, month, day, hour, 0, 0) as dt
                SELECT count()
                FROM xip.tick_fstore_hour
                WHERE dt >= '{start_dt}' AND dt <= '{end_dt}'
            """
        cnt: int = self.db_client.command(sql)
        self.logger.info(f"number of possible requests: {cnt}")
        self.logger.info(f"requests sampling: {sampling_rate}")

        sql = f"""
                WITH makeDateTime64(year, month, day, hour, 0, 0) as clk_dt
                SELECT
                    toString(clk_dt) as ts,
                    toString(addHours(clk_dt, 1)) as label_ts,
                    cpair, toString(clk_dt) as dt
                FROM xip.tick_fstore_hour
                WHERE (cityHash64(clk_dt) % {int(1.0 / sampling_rate)}) == 0
                        AND clk_dt >= '{start_dt}'
                        AND clk_dt <= '{end_dt}'
                ORDER BY (cpair, year, month, day, hour)
                """
        requests: pd.DataFrame = self.db_client.query_df(sql)
        requests["ts"] = pd.to_datetime(requests["ts"]).astype(int) // 10**9
        requests["label_ts"] = pd.to_datetime(requests["label_ts"]).astype(int) // 10**9

        min_ts = requests["ts"].min()
        requests["ts"] = (requests["ts"] - min_ts) // 3600
        requests["label_ts"] = (requests["label_ts"] - min_ts) // 3600
        requests = requests.sort_values(by=["ts", "cpair"])

        if max_num > 0 and max_num < len(requests):
            requests = requests[:max_num]

        self.logger.info(f"Got of {len(requests)}x of requests")
        return requests


class TickRalfV2TestPrepareWorker(TickRalfV2PrepareWorker):
    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests(max_num=self.max_requests)

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests


class TickRalfV2SimMedianPrepareWorker(TickRalfV2PrepareWorker):
    def create_dataset(self) -> pd.DataFrame:
        return self.create_dataset_simmedian_helper(ref_task="tickralfv2")
