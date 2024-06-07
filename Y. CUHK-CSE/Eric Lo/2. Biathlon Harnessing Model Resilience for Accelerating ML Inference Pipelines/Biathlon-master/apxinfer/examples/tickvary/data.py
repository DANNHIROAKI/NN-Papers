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


class TickVaryDataIngestor(XIPDataIngestor):
    def __init__(
        self,
        dsrc_type: str,
        dsrc: str,
        database: str,
        table: str,
        nparts: int,
        seed: int,
        year_months: list[str] = ["2022-2"],
    ) -> None:
        super().__init__(dsrc_type, dsrc, database, table, nparts, seed)
        self.year_months = year_months

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
        aux_table = "tickvary_aux"
        nrows = self.create_aux_table(aux_table)
        print(f"nrows = {nrows}")

        # count number of rows in the auxiliary table with self.monthes
        # sql = f"""
        #     SELECT count() FROM {self.database}.{aux_table}
        #     WHERE toYear(tick_dt) == 2022
        #         AND toMonth(tick_dt) in ({','.join(self.year_months)})
        # """
        # nrows = self.db_client.command(sql)
        # print(f'nrows to ingest = {nrows}')

        tb_prefix, tb_id, tb_nparts = self.table.split("_")
        tb_nparts = int(tb_nparts)
        assert tb_nparts == self.nparts
        tb_id = int(tb_id)
        prev_id = tb_id - 1
        while prev_id > 0:
            prev_table = f"{tb_prefix}_{prev_id}_{tb_nparts}"
            if DBHelper.table_exists(self.db_client, self.database, prev_table):
                if not DBHelper.table_empty(self.db_client, self.database, prev_table):
                    break
            prev_id -= 1
        prev_table = f"{tb_prefix}_{prev_id}_{tb_nparts}"
        if prev_id > 0:
            # load data from the previous table to the current table
            sql = f"""
                INSERT INTO {self.database}.{self.table}
                SELECT *
                FROM {self.database}.{prev_table}
            """
            self.db_client.command(sql)

        remaining = [self.year_months[i] for i in range(prev_id, len(self.year_months))]
        # we then insert the remaining data into the main table
        self.logger.info(f"Ingesting data from {aux_table} into table {self.table}")
        for year_month in remaining:
            year, month = year_month.split("-")
            for day in range(1, 32):
                # count the current number of rows in the main table
                sql = f"""
                    SELECT count() FROM {self.database}.{self.table}
                """
                current_count = self.db_client.command(sql)

                # count this part first
                sql = f"""
                    SELECT count() FROM {self.database}.{aux_table}
                    WHERE toYear(tick_dt) == {year}
                        AND toMonth(tick_dt) == {month}
                        AND toDayOfMonth(tick_dt) == {day}
                """
                nrows_part = self.db_client.command(sql)

                sql = f"""
                    INSERT INTO {self.database}.{self.table}
                    SELECT tmp1.*, tmp2.pid
                    FROM
                    (
                        SELECT rowNumberInAllBlocks() + {current_count} as txn_id, *
                        FROM {self.database}.{aux_table}
                        WHERE toYear(tick_dt) == {year}
                            AND toMonth(tick_dt) == {month}
                            AND toDayOfMonth(tick_dt) == {day}
                    ) as tmp1
                    JOIN
                    (
                        SELECT rowNumberInAllBlocks() + {current_count} as txn_id,
                                value % {self.nparts} as pid
                        FROM generateRandom('value UInt32', {self.seed})
                        LIMIT {nrows_part}
                    ) as tmp2
                    ON tmp1.txn_id = tmp2.txn_id
                """
                self.db_client.command(sql)

    def drop_table(self) -> None:
        DBHelper.drop_table(self.db_client, self.database, self.table)
        self.drop_aux_table(f"{self.table}_aux")

    def clear_table(self) -> None:
        DBHelper.clear_table(self.db_client, self.database, self.table)
        self.clear_aux_table(f"{self.table}_aux")


class TickVaryHourFStoreIngestor(XIPDataIngestor):
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


def ingest(nparts: int = 100, seed: int = 0,
           num_months: int = 1, verbose: bool = False):
    """
    # first 7 months
    12G     ./February2022 (275M)
    9.7G    ./March2022
    2.8G    ./April2022
    3.3G    ./May2022
    4.0G    ./June2022
    5.7G    ./July2022
    12G     ./August2022

    # other months
    2.2G    ./January2022

    2.0G    ./December2021
    1.7G    ./November2021
    2.1G    ./October2021
    2.1G    ./September2021
    1.8G    ./August2021
    2.3G    ./July2021
    2.1G    ./June2021
    2.5G    ./March2021
    1.8G    ./April2021
    2.0G    ./May2021
    1.9G    ./February2021
    2.0G    ./January2021

    2.0G    ./December2020
    2.5G    ./November2020
    2.4G    ./October2020
    2.9G    ./September2020
    2.9G    ./August2020
    3.4G    ./July2020
    3.3G    ./June2020
    3.3G    ./April2020
    2.1G    ./May2020
    99G     .
    """
    all_months_dirs = ["February2022", "March2022", "April2022", "May2022", "June2022", "July2022", "August2022"]
    all_months_dirs += ["January2022"]
    all_months_dirs += ["December2021", "November2021", "October2021", "September2021", "August2021", "July2021", "June2021", "March2021", "April2021", "May2021", "February2021", "January2021"]
    all_months_dirs += ["December2020", "November2020", "October2020", "September2020", "August2020", "July2020", "June2020", "April2020", "May2020"]
    date_list = ["2022-2", "2022-3", "2022-4", "2022-5", "2022-6", "2022-7", "2022-8"]
    date_list += ["2022-1"]
    date_list += ["2021-12", "2021-11", "2021-10", "2021-9", "2021-8", "2021-7", "2021-6", "2021-3", "2021-4", "2021-5", "2021-2", "2021-1"]
    date_list += ["2020-12", "2020-11", "2020-10", "2020-9", "2020-8", "2020-7", "2020-6", "2020-4", "2020-5"]

    dsrc_type = "user_files_dir"
    possible_dsrcs = [
        "/public/ckchang/db/clickhouse/user_files/tick-data",
        "/mnt/sdb/dataset/tick-data",
        "/mnt/hddraid/clickhouse-data/user_files/tick-data",
        "/var/lib/clickhouse/user_files/tick-data",
    ]
    dsrc = None
    for src in possible_dsrcs:
        if os.path.exists(src):
            dsrc = src
            print(f"dsrc path: {dsrc}")
            break
    if dsrc is None:
        raise RuntimeError("no valid dsrc!")
    # check if the data directory exists
    for month in all_months_dirs[:num_months]:
        assert os.path.exists(os.path.join(dsrc, month)), f"{month} does not exist!"

    month_ids = [date_list[i] for i in range(num_months)]
    ingestor = TickVaryDataIngestor(
        dsrc_type=dsrc_type,
        dsrc=dsrc,
        database=f"xip_{seed}",
        table=f"tickvary_{num_months}_{nparts}",
        nparts=nparts,
        seed=seed,
        year_months=month_ids
    )
    ingestor.run()

    ingestor = TickVaryHourFStoreIngestor(
        dsrc_type="clickhouse",
        # dsrc=f"xip.tickvary_{nparts}",
        dsrc=f"xip_{seed}.tickvary_aux",
        database=f"xip_{seed}",
        table="tickvary_fstore_hour",
        nparts=nparts,
        seed=seed,
    )
    ingestor.run()
