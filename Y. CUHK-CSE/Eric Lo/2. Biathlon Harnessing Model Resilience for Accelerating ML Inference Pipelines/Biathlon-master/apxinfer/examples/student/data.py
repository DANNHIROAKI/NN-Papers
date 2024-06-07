""" column's unique values and type
    session_id           23562  int64
    index                20348  int64
    elapsed_time       5042639  int64
    event_name              11  str
    name                     6  str
    level                   23  int64
    page                     7  float64
    room_coor_x       12538215  float64
    room_coor_y        9551136  float64
    screen_coor_x        57477  float64
    screen_coor_y       102591  float64
    hover_duration       24101  float64
    text                   597  str
    fqid                   128  str
    room_fqid               19  str
    text_fqid              126  str
    fullscreen               2  bool
    hq                       2  bool
    music                    2  bool
    level_group              3  str
"""

from typing import List
import numpy as np
import pandas as pd
import datetime as dt
import os

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


class StudentRequest(XIPRequest):
    req_session_id: int
    req_qno: int


def get_dsrc():
    possible_dsrcs = [
        "/public/ckchang/db/clickhouse/user_files/predict-student-performance-from-game-play",
        "/mnt/sdb/dataset/predict-student-performance-from-game-play",
        "/mnt/hddraid/clickhouse-data/user_files/predict-student-performance-from-game-play",
        "/var/lib/clickhouse/user_files/predict-student-performance-from-game-play",
    ]
    dsrc = None
    for src in possible_dsrcs:
        if os.path.exists(src):
            dsrc = src
            print(f"dsrc path: {dsrc}")
            break
    if dsrc is None:
        raise RuntimeError("no valid dsrc!")
    return dsrc


class StudentIngestor(XIPDataIngestor):
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
        # session_id,index,elapsed_time,event_name,name,level,page,room_coor_x,room_coor_y,screen_coor_x,screen_coor_y,hover_duration,text,fqid,room_fqid,text_fqid,fullscreen,hq,music,level_group
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
            `txn_id` UInt64,
            `session_id` UInt64,
            `index` UInt64,
            `elapsed_time` UInt64,
            `event_name` String,
            `name` String,
            `level` UInt64,
            `page` Nullable(Float64),
            `room_coor_x` Nullable(Float64),
            `room_coor_y` Nullable(Float64),
            `screen_coor_x` Nullable(Float64),
            `screen_coor_y` Nullable(Float64),
            `hover_duration` Nullable(Float64),
            `text` Nullable(String),
            `fqid` Nullable(String),
            `room_fqid` Nullable(String),
            `text_fqid` Nullable(String),
            `fullscreen` UInt8,
            `hq` UInt8,
            `music` UInt8,
            `level_group` String,
            `pid` UInt32 -- partition key, used for sampling
        ) ENGINE = MergeTree()
        PARTITION BY `pid`
        ORDER BY (`session_id`, `level_group`)
        -- SETTINGS index_granularity = 32
        """
        self.db_client.command(sql)

    def create_aux_table(self, aux_table: str) -> int:
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{aux_table} (
                `session_id` UInt64,
                `index` UInt64,
                `elapsed_time` UInt64,
                `event_name` String,
                `name` String,
                `level` UInt64,
                `page` Nullable(Float64),
                `room_coor_x` Nullable(Float64),
                `room_coor_y` Nullable(Float64),
                `screen_coor_x` Nullable(Float64),
                `screen_coor_y` Nullable(Float64),
                `hover_duration` Nullable(Float64),
                `text` Nullable(String),
                `fqid` Nullable(String),
                `room_fqid` Nullable(String),
                `text_fqid` Nullable(String),
                `fullscreen` UInt8,
                `hq` UInt8,
                `music` UInt8,
                `level_group` String
            ) ENGINE = MergeTree()
            ORDER BY (`session_id`, `index`)
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
                SELECT rowNumberInAllBlocks() as txn_id, value % {self.nparts} as pid
                FROM generateRandom('value UInt32', {self.seed})
                LIMIT {nrows}
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
    txns_src = "file('predict-student-performance-from-game-play/train.csv', CSVWithNames)"
    txns_ingestor = StudentIngestor(
        dsrc_type="user_files",
        dsrc=txns_src,
        database=f"xip_{seed}",
        table=f"student_{nparts}",
        nparts=nparts,
        seed=seed,
    )
    txns_ingestor.run()


def db_migration(from_dbtable: str, to_dbtable: str, session_ids: list):
    db_client = DBHelper.get_db_client()
    sql = f"""
        CREATE TABLE IF NOT EXISTS {to_dbtable}
        ENGINE = MergeTree()
        PARTITION BY `pid`
        ORDER BY (`session_id`, `level_group`)
        AS SELECT *
        FROM {from_dbtable}
        WHERE session_id IN ({', '.join(map(str, session_ids))})
        """
    db_client.command(sql)


def db_migration_v2(from_dbtable: str, to_dbtable: str,
                    session_ids: list,
                    level_group: str):
    db_client = DBHelper.get_db_client()
    sql = f"""
        CREATE TABLE IF NOT EXISTS {to_dbtable}
        ENGINE = MergeTree()
        PARTITION BY `pid`
        ORDER BY (`session_id`, `level_group`)
        AS SELECT *
        FROM {from_dbtable}
        WHERE level_group = '{level_group}' 
            AND session_id IN ({', '.join(map(str, session_ids))})
        """
    db_client.command(sql)


if __name__ == "__main__":
    ingest()
