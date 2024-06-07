from typing import List
import numpy as np
import pandas as pd
import datetime as dt
import os
from tqdm import tqdm

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


class TickRequest(XIPRequest):
    req_dt: str
    req_cpair: str


def get_all_files(data_dir: str) -> List[str]:
    # initialize an empty list to store the file paths
    file_paths = []

    # use os.walk to search for files recursively
    for dirpath, dirnames, filenames in os.walk(data_dir):
        # iterate over the filenames and add the full file path to the list
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)
    return file_paths


class TickDataIngestor(XIPDataIngestor):
    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
            tick_id UInt64,
            cpair String,
            tick_dt DateTime64,
            bid Float32,
            ask Float32,
            pid UInt32
        ) ENGINE = MergeTree()
        PARTITION BY pid
        ORDER BY (cpair, tick_dt)
        SETTINGS index_granularity = 32
        """
        self.db_client.command(sql)

    def create_aux_table(self, aux_table: str) -> int:
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{aux_table} (
            cpair String,
            tick_dt DateTime64,
            bid Float32,
            ask Float32
        ) ENGINE = MergeTree()
        PARTITION BY cpair
        ORDER BY tick_dt
        """
        self.db_client.command(sql)
        if DBHelper.table_empty(self.db_client, self.database, aux_table):
            self.logger.info(f"Ingesting data from {self.dsrc} into table {aux_table}")
            assert self.dsrc_type == "user_files_dir"
            files = get_all_files(self.dsrc)
            for fpath in tqdm(files, desc="ingesting into aux", total=len(files)):
                sql = f""" INSERT INTO {self.database}.{aux_table} \
                        SELECT cpair , parseDateTime64BestEffort(dt_str) as tick_dt, \
                                bid, ask \
                        FROM input('cpair String, dt_str String, \
                            bid Float32, ask Float32') \
                        FORMAT CSV \
                """
                command = f"""clickhouse-client --query "{sql}" < {fpath}"""
                os.system(command)
        return DBHelper.get_table_size(self.db_client, self.database, aux_table)

    def ingest_data(self) -> None:
        self.logger.info(f"Ingesting data from {self.dsrc} into table {self.table}")
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        assert (
            self.dsrc_type == "user_files_dir"
        ), f"Unsupported data source type {self.dsrc_type}"

        # we first create an auxiliary table to store the data
        # aux_table = f"{self.table}_aux"
        aux_table = "tick_aux"
        nrows = self.create_aux_table(aux_table)
        print(f"nrows = {nrows}")

        # we then insert the data into the main table
        self.logger.info(f"Ingesting data from {aux_table} into table {self.table}")
        sql = f"""
            INSERT INTO {self.database}.{self.table}
            SELECT tmp1.*, tmp2.pid
            FROM
            (
                SELECT rowNumberInAllBlocks() as txn_id, *
                FROM {self.database}.{aux_table}
            ) as tmp1
            JOIN
            (
                SELECT rowNumberInAllBlocks() as txn_id,
                        value % {self.nparts} as pid
                FROM generateRandom('value UInt32', {self.seed})
                LIMIT {nrows}
            ) as tmp2
            ON tmp1.txn_id = tmp2.txn_id
        """
        if nrows > 1e8:
            nrows = int(1e8)
            sql = f"""
                INSERT INTO {self.database}.{self.table}
                SELECT tmp1.*, tmp2.pid
                FROM
                (
                    SELECT rowNumberInAllBlocks() as txn_id, *
                    FROM {self.database}.{aux_table}
                ) as tmp1
                JOIN
                (
                    SELECT rowNumberInAllBlocks() as txn_id,
                            value % {self.nparts} as pid
                    FROM generateRandom('value UInt32', {self.seed})
                    LIMIT {nrows}
                ) as tmp2
                ON (tmp1.txn_id % {nrows}) = tmp2.txn_id
            """

        self.db_client.command(sql)

        # we drop the auxiliary table
        # self.drop_aux_table(aux_table)

    def drop_aux_table(self, aux_table: str) -> None:
        DBHelper.drop_table(self.db_client, self.database, aux_table)

    def drop_table(self) -> None:
        DBHelper.drop_table(self.db_client, self.database, self.table)
        self.drop_aux_table(f"{self.table}_aux")

    def clear_aux_table(self, aux_table: str) -> None:
        DBHelper.clear_table(self.db_client, self.database, aux_table)

    def clear_table(self) -> None:
        DBHelper.clear_table(self.db_client, self.database, self.table)
        self.clear_aux_table(f"{self.table}_aux")


class TickHourFStoreIngestor(XIPDataIngestor):
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
        self.agg_ops = [
            "count",
            "avg",
            "sum",
            "varPop",
            "stddevPop",
            "median",
            "min",
            "max",
        ]
        self.agg_cols = ["bid", "ask"]
        self.time_keys = ["year", "month", "day", "hour"]
        self.time_ops = ["toYear", "toMonth", "toDayOfMonth", "toHour"]

    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return
        aggs = [f"{agg}_{col} Float32" for col in self.agg_cols for agg in self.agg_ops]
        key_cols = [f"{key} Int32" for key in self.time_keys]
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                cpair String,
                {', '.join(key_cols)},
                {', '.join(aggs)}
            ) ENGINE = MergeTree()
            PARTITION BY cpair
            ORDER BY ({', '.join(self.time_keys)})
            """
        self.db_client.command(sql)

    def ingest_data(self) -> None:
        self.logger.info(
            f"Ingesting data from {self.dsrc} into table {self.database}.{self.table}"
        )
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        assert (
            self.dsrc_type == "clickhouse"
        ), f"Unsupported data source type {self.dsrc_type}"
        aggs = [
            f"{agg}({col}) AS {agg}_{col}"
            for col in self.agg_cols
            for agg in self.agg_ops
        ]
        assert len(self.time_ops) == len(self.time_keys)
        time_cols = [
            f"{self.time_ops[i]}(tick_dt) as {self.time_keys[i]}"
            for i in range(len(self.time_keys))
        ]
        sql = (
            f"""select distinct (toYear(tick_dt), toMonth(tick_dt)) from {self.dsrc}"""
        )
        all_rounds = self.db_client.command(sql).split("\n")
        print(f"all_rounds={all_rounds}")
        for round in tqdm(all_rounds, desc="ingesting by month", total=len(all_rounds)):
            year, month = round[1:-1].split(",")
            sql = f"""
                INSERT INTO {self.database}.{self.table}
                SELECT cpair, {', '.join(time_cols)}, {', '.join(aggs)}
                FROM (SELECT *
                    FROM {self.dsrc}
                    WHERE toYear(tick_dt) = '{year}'
                        AND toMonth(tick_dt) = '{month}'
                    )
                GROUP BY cpair, {', '.join(self.time_keys)}
            """
            self.db_client.command(sql)

        # # the following code could crash due to OOM
        # sql = f"""
        #     INSERT INTO {self.database}.{self.table}
        #     SELECT cpair, {', '.join(time_cols)},
        #         {', '.join(aggs)}
        #     FROM {self.dsrc}
        #     GROUP BY cpair, {', '.join(self.time_keys)}
        # """
        # self.db_client.command(sql)


def ingest(nparts: int = 100, seed: int = 0, verbose: bool = False):
    dsrc_type = "user_files_dir"
    possible_dsrcs = [
        "/public/ckchang/db/clickhouse/user_files/tick-data/February2022",
        "/mnt/sdb/dataset/tick-data/February2022",
        "/mnt/hddraid/clickhouse-data/user_files/tick-data/February2022",
        "/var/lib/clickhouse/user_files/tick-data/February2022",
    ]
    dsrc = None
    for src in possible_dsrcs:
        if os.path.exists(src):
            dsrc = src
            print(f"dsrc path: {dsrc}")
            break
    if dsrc is None:
        raise RuntimeError("no valid dsrc!")

    ingestor = TickDataIngestor(
        dsrc_type=dsrc_type,
        dsrc=dsrc,
        database=f"xip_{seed}",
        table=f"tick_{nparts}",
        nparts=nparts,
        seed=seed,
    )
    ingestor.run()

    ingestor = TickHourFStoreIngestor(
        dsrc_type="clickhouse",
        dsrc=f"xip_{seed}.tick_{nparts}",
        database=f"xip_{seed}",
        table="tick_fstore_hour",
        nparts=nparts,
        seed=seed,
    )
    ingestor.run()
