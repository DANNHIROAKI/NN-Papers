import pandas as pd
import os
from typing import Tuple
from tqdm import tqdm

from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker


class TripsPrepareWorker(XIPPrepareWorker):
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
        self.table = f"trips_{nparts}"

    def get_requests(self) -> pd.DataFrame:
        trips_from = "2015-08-01 00:00:00"
        trips_to = "2015-08-15 00:00:00"
        total_num = self.db_client.command(
            f""" SELECT count()
                FROM {self.database}.{self.table}
                WHERE pickup_datetime >= '{trips_from}'
                      AND pickup_datetime < '{trips_to}'
            """
        )
        self.logger.info(f"Total number of trips: {total_num}")
        # as there is too much trips (around 5M), we should sample some of them
        sampling_rate = 0.01
        sql = f"""
                SELECT
                    trip_id, toString(pickup_datetime) as pickup_datetime,
                    pickup_ntaname, dropoff_ntaname,
                    pickup_longitude, pickup_latitude,
                    dropoff_longitude, dropoff_latitude,
                    passenger_count, trip_distance
                FROM {self.database}.{self.table}
                WHERE {self.database}.{self.table}.pickup_datetime >= '{trips_from}'
                        AND {self.database}.{self.table}.pickup_datetime < '{trips_to}'
                        AND intHash64(trip_id) % ({int(1.0/sampling_rate)}) == 0
                ORDER BY pickup_datetime
                """
        requests: pd.DataFrame = self.db_client.query_df(sql)
        # drop requests with invalid pickup/dropoff locations or ntaname is ''
        before_size = len(requests)
        requests = requests[requests["pickup_ntaname"] != ""]
        requests = requests[requests["dropoff_ntaname"] != ""]
        self.logger.info(
            f"Dropped {before_size - len(requests)}x of requests with invalid locations"
        )

        if self.max_requests > 0 and self.max_requests < len(requests):
            # select max_requests in the middle
            start = int((len(requests) - self.max_requests) / 2)
            end = start + self.max_requests
            requests = requests.iloc[start:end]
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")
        trip_ids = [str(x) for x in requests["req_trip_id"].values]
        sql = f"""
            SELECT trip_id, fare_amount
            FROM {self.database}.{self.table}
            WHERE trip_id IN ({','.join(trip_ids)})
            """
        df: pd.DataFrame = self.db_client.query_df(sql)
        labels_pds = requests.merge(
            df, left_on="req_trip_id", right_on="trip_id", how="left"
        )["fare_amount"]
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds


class TripsRalfPrepareWorker(TripsPrepareWorker):
    def _extract_requests(
        self,
        trips_from: str = "2015-08-01 00:00:00",
        trips_to: str = "2015-08-15 00:00:00",
        sampling_rate: float = 1,
        max_num: int = 0,
    ) -> pd.DataFrame:
        total_num = self.db_client.command(
            f""" SELECT count()
                FROM {self.database}.{self.table}
                WHERE pickup_datetime >= '{trips_from}'
                      AND pickup_datetime < '{trips_to}'
            """
        )
        self.logger.info(f"Total number of trips: {total_num}")
        # as there is too much trips (around 5M), we should sample some of them
        sql = f"""
                SELECT
                    toString(pickup_datetime) as ts,
                    toString(dropoff_datetime) as label_ts,
                    trip_id,
                    toString(pickup_datetime) as pickup_datetime,
                    pickup_ntaname, dropoff_ntaname,
                    pickup_longitude, pickup_latitude,
                    dropoff_longitude, dropoff_latitude,
                    passenger_count, trip_distance
                FROM {self.database}.{self.table}
                WHERE {self.database}.{self.table}.pickup_datetime >= '{trips_from}'
                        AND {self.database}.{self.table}.pickup_datetime < '{trips_to}'
                        AND intHash64(trip_id) % ({int(1.0/sampling_rate)}) == 0
                ORDER BY pickup_datetime, pickup_ntaname, dropoff_ntaname, trip_id
                """
        requests: pd.DataFrame = self.db_client.query_df(sql)
        # drop requests with invalid pickup/dropoff locations or ntaname is ''
        before_size = len(requests)
        requests = requests[requests["pickup_ntaname"] != ""]
        requests = requests[requests["dropoff_ntaname"] != ""]
        self.logger.info(
            f"Dropped {before_size - len(requests)}x of requests with invalid locations"
        )
        requests["ts"] = pd.to_datetime(requests["ts"]).astype(int) // 10**9
        requests["label_ts"] = pd.to_datetime(requests["label_ts"]).astype(int) // 10**9

        if max_num > 0 and max_num < len(requests):
            requests = requests[:max_num]

        return requests

    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests()

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")
        trip_ids = [str(x) for x in requests["req_trip_id"].values]
        df = None
        for i in tqdm(range(0, len(trip_ids), 1000)):
            sql = f"""
                SELECT trip_id, fare_amount
                FROM {self.database}.{self.table}
                WHERE trip_id IN ({','.join(trip_ids[i:i+1000])})
                """
            if df is None:
                df: pd.DataFrame = self.db_client.query_df(sql)
            else:
                df = pd.concat([df, self.db_client.query_df(sql)])
        labels_pds = requests.merge(
            df, left_on="req_trip_id", right_on="trip_id", how="left"
        )["fare_amount"]
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds

    def split_dataset(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n_train = len(dataset) // 2
        n_valid = 100
        train_set = dataset[:n_train]
        test_set = dataset[n_train:]
        valid_set = test_set[:n_valid]
        return train_set, valid_set, test_set


class TripsRalfTestPrepareWorker(TripsRalfPrepareWorker):
    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests(max_num=self.max_requests)

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests


class TripsRalf2HPrepareWorker(TripsRalfPrepareWorker):
    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests(
            trips_from="2015-08-01 00:00:00", trips_to="2015-08-01 02:00:00"
        )

        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests


class TripsRalfV2PrepareWorker(TripsRalfPrepareWorker):
    def get_requests(self) -> pd.DataFrame:
        # part1 without sampling: 2632114
        part1 = self._extract_requests(
            trips_from="2015-08-01 00:00:00",
            trips_to="2015-08-08 00:00:00",
            sampling_rate=0.001,
        )
        # around 2.6k in part1

        part2 = self._extract_requests(
            trips_from="2015-08-08 00:00:00",
            trips_to="2015-08-08 01:00:00",
            sampling_rate=1,
        )

        requests = pd.concat([part1, part2])
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def split_dataset(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame]:
        split_ts = pd.to_datetime("2015-08-08 00:00:00").value // 10**9
        train_set = dataset[dataset["req_ts"] < split_ts]
        test_set = dataset[dataset["req_ts"] >= split_ts]
        valid_set = test_set[:100]
        return train_set, valid_set, test_set


class TripsRalfV3PrepareWorker(TripsRalfPrepareWorker):
    def get_requests(self) -> pd.DataFrame:
        # part1 without sampling: 2632114
        part1 = self._extract_requests(
            trips_from="2015-08-01 00:00:00",
            trips_to="2015-08-08 00:00:00",
            sampling_rate=0.001,
        )
        # around 2.6k in part1

        part2 = self._extract_requests(
            trips_from="2015-08-08 00:00:00",
            trips_to="2015-08-08 00:05:00",
            sampling_rate=1,
        )

        requests = pd.concat([part1, part2])
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def split_dataset(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame]:
        split_ts = pd.to_datetime("2015-08-08 00:00:00").value // 10**9
        train_set = dataset[dataset["req_ts"] < split_ts]
        test_set = dataset[dataset["req_ts"] >= split_ts]
        valid_set = test_set[:100]
        return train_set, valid_set, test_set


class TripsRalfV2SimMedianPrepareWorker(TripsRalfV2PrepareWorker):
    def create_dataset(self) -> pd.DataFrame:
        return self.create_dataset_simmedian_helper(ref_task="tripsralfv2")


class TripsRalfV3SimMedianPrepareWorker(TripsRalfV3PrepareWorker):
    def create_dataset(self) -> pd.DataFrame:
        return self.create_dataset_simmedian_helper(ref_task="tripsralfv3")
