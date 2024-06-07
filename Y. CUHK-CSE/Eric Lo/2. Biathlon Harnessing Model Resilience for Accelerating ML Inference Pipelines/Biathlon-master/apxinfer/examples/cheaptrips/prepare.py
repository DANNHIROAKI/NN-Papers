import pandas as pd
import os

from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker


class CheapTripsPrepareWorker(XIPPrepareWorker):
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
        nparts: int
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
            SELECT trip_id, (fare_amount <= 10) as cheap_trip
            FROM {self.database}.{self.table}
            WHERE trip_id IN ({','.join(trip_ids)})
            """
        df: pd.DataFrame = self.db_client.query_df(sql)
        labels_pds = requests.merge(
            df, left_on="req_trip_id", right_on="trip_id", how="left"
        )["cheap_trip"]
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds
