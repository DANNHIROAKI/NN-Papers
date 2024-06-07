from apxinfer.core.utils import XIPRequest
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader
import os


class TripsRequest(XIPRequest):
    req_trip_id: int
    req_pickup_datetime: str
    req_pickup_ntaname: str
    req_dropoff_ntaname: str
    req_pickup_longitude: float
    req_pickup_latitude: float
    req_dropoff_longitude: float
    req_dropoff_latitude: float
    req_passenger_count: int
    req_trip_distance: float


class TripsIngestor(XIPDataIngestor):
    def __init__(
        self,
        dsrc_type: str,
        dsrc: str,
        database: str,
        table: str,
        nparts: int,
        seed: int,
    ) -> None:
        super().__init__(dsrc_type, dsrc, database, table, nparts, seed)

    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return
        sql = f"""CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                    trip_id UInt32,
                    pickup_datetime DateTime64,
                    dropoff_datetime DateTime64,
                    pickup_longitude Float64 DEFAULT 0.0,
                    pickup_latitude Float64 DEFAULT 0.0,
                    dropoff_longitude Float64 DEFAULT 0.0,
                    dropoff_latitude Float64 DEFAULT 0.0,
                    passenger_count UInt8,
                    trip_distance Float32,
                    fare_amount Float32,
                    extra Float32,
                    tip_amount Float32,
                    tolls_amount Float32,
                    total_amount Float32,
                    payment_type Enum(
                        'CSH' = 1,
                        'CRE' = 2,
                        'NOC' = 3,
                        'DIS' = 4,
                        'UNK' = 5
                    ),
                    pickup_ntaname LowCardinality(String),
                    dropoff_ntaname LowCardinality(String),
                    trip_duration Float32,
                    pid UInt32, -- partition key, used for sampling
                    INDEX idx0 dropoff_datetime TYPE minmax GRANULARITY 1,
                    INDEX idx1 pickup_ntaname TYPE set(0) GRANULARITY 1,
                    INDEX idx2 dropoff_ntaname TYPE set(0) GRANULARITY 1,
                    INDEX idx3 passenger_count TYPE set(0) GRANULARITY 1
                ) ENGINE = MergeTree()
                PARTITION BY pid
                ORDER BY pickup_datetime
                SETTINGS index_granularity = 1024
                        , min_rows_for_wide_part = 0
                        , min_bytes_for_wide_part = 0
            """
        self.db_client.command(sql)

    def create_dsrc(self, dtable: str):
        sql_create = f"""
            -- create tables for trips in clickhouse
            CREATE TABLE {dtable} (
                trip_id UInt32,
                pickup_datetime DateTime,
                dropoff_datetime DateTime,
                pickup_longitude Float64 DEFAULT 0.0,
                pickup_latitude Float64 DEFAULT 0.0,
                dropoff_longitude Float64 DEFAULT 0.0,
                dropoff_latitude Float64 DEFAULT 0.0,
                passenger_count UInt8,
                trip_distance Float32,
                fare_amount Float32,
                extra Float32,
                tip_amount Float32,
                tolls_amount Float32,
                total_amount Float32,
                payment_type Enum(
                    'CSH' = 1,
                    'CRE' = 2,
                    'NOC' = 3,
                    'DIS' = 4,
                    'UNK' = 5
                ),
                pickup_ntaname LowCardinality(String),
                dropoff_ntaname LowCardinality(String)
            ) ENGINE = MergeTree PARTITION BY passenger_count
            ORDER BY (pickup_datetime, dropoff_datetime)
            """

        dsrc_home = "/var/lib/clickhouse/user_files/taxi-2015/trips_{0..19}.gz"
        sql_insert = f"""
            -- insert data into trips from lcoal files (20m records)
            INSERT INTO {dtable}
            SELECT * FROM file('{dsrc_home}', TSVWithNames)
            -- FROM INFILE '{dsrc_home}' FORMAT TSVWithNames
            """

        # possible_dsrcs = [
        #     "/public/ckchang/db/clickhouse/user_files/taxi-2015/trips_{0..19}.gz",
        #     "/var/lib/clickhouse/user_files/taxi-2015/trips_{0..19}.gz",
        # ]
        # dsrc = None
        # for src in possible_dsrcs:
        #     if os.path.exists(src):
        #         dsrc = src
        #         print(f"dsrc path: {dsrc}")
        #         break
        # if dsrc is None:
        #     raise RuntimeError("no valid dsrc!")

        # sql_insert = f"""
        #     -- insert data into trips from lcoal files (20m records)
        #     INSERT INTO trips
        #     FROM INFILE {dsrc} FORMAT TSVWithNames;
        # """
        sql_alter = f"""
            -- add new column trip_duration as (dropoff_datetime - pickup_datetime)
            ALTER TABLE {dtable}
            ADD COLUMN trip_duration Float32
            """
        sql_update = f"""
            -- update trip_duration
            ALTER TABLE {dtable}
            UPDATE trip_duration = (dropoff_datetime - pickup_datetime)
            WHERE 1
            """
        sql_clean = f"""
            -- clean the data.
            -- remove records with negative trip_duration, trip_distance, fare_amount,
            -- total_amount, and passenger_count
            ALTER TABLE {dtable} DELETE
            WHERE trip_duration < 0
                OR trip_distance < 0
                OR fare_amount < 0
                OR extra < 0
                OR tip_amount < 0
                OR total_amount < 0
                OR passenger_count < 0
            """
        self.db_client.command(sql_create)
        self.db_client.command(sql_insert)
        self.db_client.command(sql_alter)
        self.db_client.command(sql_update)
        self.db_client.command(sql_clean)

    def ingest_data(self) -> None:
        assert (
            self.dsrc_type == "clickhouse"
        ), f"Unsupported data source type {self.dsrc_type}"

        self.logger.info(
            f"Ingesting data from {self.dsrc} into table {self.database}.{self.table}"
        )

        if not DBHelper.table_exists(
            self.db_client, self.dsrc.split(".")[0], self.dsrc.split(".")[1]
        ):
            self.create_dsrc(self.dsrc)
        elif DBHelper.table_empty(
            self.db_client, self.dsrc.split(".")[0], self.dsrc.split(".")[1]
        ):
            self.create_dsrc(self.dsrc)

        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        nrows = DBHelper.get_table_size(
            self.db_client, self.dsrc.split(".")[0], self.dsrc.split(".")[1]
        )
        sql = f"""
            INSERT INTO {self.database}.{self.table}
            SETTINGS max_partitions_per_insert_block = 1000
            SELECT trip_id, pickup_datetime, dropoff_datetime,
              pickup_longitude, pickup_latitude,
              dropoff_longitude, dropoff_latitude,
              passenger_count, trip_distance, fare_amount,
              extra, tip_amount, tolls_amount, total_amount,
              payment_type, pickup_ntaname, dropoff_ntaname,
              trip_duration, pid
            FROM
            (
                SELECT *, rowNumberInAllBlocks() as row_id
                FROM {self.dsrc}
            ) as tmp1
            JOIN
            (
                SELECT value % {self.nparts} as pid, rowNumberInAllBlocks() as row_id
                FROM (
                    SELECT *
                    FROM generateRandom('value UInt32', {self.seed})
                    LIMIT {nrows}
                )
            ) as tmp2
            ON tmp1.row_id = tmp2.row_id
        """
        self.db_client.command(sql)


def get_ingestor(nparts: int = 100, seed: int = 0):
    ingestor = TripsIngestor(
        dsrc_type="clickhouse",
        dsrc="default.trips",
        database=f"xip_{seed}",
        table=f"trips_{nparts}",
        nparts=nparts,
        seed=seed,
    )
    return ingestor


def get_dloader(nparts: int = 100, seed: int = 0,
                verbose: bool = False) -> XIPDataLoader:
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"trips_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")
    return data_loader
