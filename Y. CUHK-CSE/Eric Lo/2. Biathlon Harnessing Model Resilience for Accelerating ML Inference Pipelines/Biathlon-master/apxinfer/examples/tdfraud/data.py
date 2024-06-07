from typing import List
import numpy as np
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


# ip,app,device,os,channel,click_time,attributed_time,is_attributed
class TDFraudRequest(XIPRequest):
    req_txn_id: int
    req_ip: int
    req_app: int
    req_device: int
    req_os: int
    req_channel: int
    req_click_time: dt.datetime


class TDFraudTxnsIngestor(XIPDataIngestor):
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
        self.db_client = DBHelper.get_db_client()

    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return
        # ip,app,device,os,channel,click_time,attributed_time,is_attributed
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
            txn_id UInt32,
            ip UInt32,
            app UInt32,
            device UInt32,
            os UInt32,
            channel UInt32,
            click_time DateTime,
            attributed_time DateTime,
            is_attributed UInt8,
            pid UInt32 -- partition key, used for sampling
        ) ENGINE = MergeTree()
        PARTITION BY pid
        ORDER BY (ip, app, os, click_time)
        SETTINGS index_granularity = 32
        """
        self.db_client.command(sql)

    def create_aux_table(self, aux_table: str) -> int:
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{aux_table} (
                ip UInt32,
                app UInt32,
                device UInt32,
                os UInt32,
                channel UInt32,
                click_time DateTime,
                attributed_time DateTime,
                is_attributed UInt8
        ) ENGINE = MergeTree()
            ORDER BY click_time
            """
        self.db_client.command(sql)
        if DBHelper.table_empty(self.db_client, self.database, aux_table):
            self.logger.info(
                f"Ingesting data from {self.dsrc} into table {aux_table} in database {self.database}"
            )
            sql = f"""
                INSERT INTO {self.database}.{aux_table}
                SELECT *
                FROM {self.dsrc}
                FORMAT CSVWithNames
                """
            self.db_client.command(sql)
            # load test set also: test_supplement.csv, which is in the same folder of train.csv
            test_data = self.dsrc.replace("train.csv", "test_supplement.csv")
            sql = f"""
                INSERT INTO {self.database}.{aux_table}
                SELECT *
                FROM {test_data}
                FORMAT CSVWithNames
                """
            self.db_client.command(sql)
        return DBHelper.get_table_size(self.db_client, self.database, aux_table)

    def ingest_data(self) -> None:
        self.logger.info(
            f"Ingesting data from {self.dsrc} into table {self.database}.{self.table}"
        )
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        assert (
            self.dsrc_type == "user_files"
        ), f"Unsupported data source type {self.dsrc_type}"

        # we first create an auxiliary table to store the data
        aux_table = f"{self.table}_aux"
        nrows = self.create_aux_table(aux_table)
        print(f"nrows = {nrows}")

        # we then insert the data into the main table
        self.logger.info(
            f"Ingesting data from {aux_table} into table {self.database}.{self.table}"
        )
        # data is to large, we ingest day by day
        # get distinct days
        sql = f"""
            SELECT DISTINCT toHour(click_time) as day
            FROM {self.database}.{aux_table}
            ORDER BY day
        """
        hours = self.db_client.query_df(sql)["day"].values
        for hour in hours:
            # get current number of rows
            sql = f"""
                SELECT count()
                FROM {self.database}.{self.table}
            """
            current_nrows = self.db_client.command(sql)

            sql = f"""
                SELECT count()
                FROM {self.database}.{aux_table}
                WHERE toHour(click_time) = '{hour}'
            """
            nrows_day = self.db_client.command(sql)

            sql = f"""
                INSERT INTO {self.database}.{self.table}
                SELECT tmp1.*, tmp2.pid
                FROM
                (
                    SELECT rowNumberInAllBlocks() + {current_nrows} as txn_id, *
                    FROM {self.database}.{aux_table}
                    WHERE toHour(click_time) = '{hour}'
                ) as tmp1
                JOIN
                (
                    SELECT rowNumberInAllBlocks() + {current_nrows} as txn_id, value % {self.nparts} as pid
                    FROM generateRandom('value UInt32', {self.seed})
                    LIMIT {nrows_day}
                ) as tmp2
                ON tmp1.txn_id = tmp2.txn_id
            """
            self.db_client.command(sql)

        # we drop the auxiliary table
        self.drop_aux_table(aux_table)

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


def ingest(nparts: int = 100, seed: int = 0, verbose: bool = False):
    txns_src = "file('talkingdata/adtracking-fraud/train.csv', CSVWithNames)"
    txns_ingestor = TDFraudTxnsIngestor(
        dsrc_type="user_files",
        dsrc=txns_src,
        database=f"xip_{seed}",
        table=f"tdfraud_{nparts}",
        nparts=nparts,
        seed=seed,
    )
    txns_ingestor.run()
