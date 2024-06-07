from typing import List
import numpy as np
import glob
import os
from tqdm import tqdm

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


class MachineryRequest(XIPRequest):
    req_bid: int


def get_raw_data_files_list(data_dir: str) -> List[str]:
    normal_file_names = glob.glob(os.path.join(data_dir, "normal", "normal", "*.csv"))
    imnormal_file_names_6g = glob.glob(
        os.path.join(data_dir, "imbalance", "imbalance", "6g", "*.csv")
    )
    imnormal_file_names_10g = glob.glob(
        os.path.join(data_dir, "imbalance", "imbalance", "10g", "*.csv")
    )
    imnormal_file_names_15g = glob.glob(
        os.path.join(data_dir, "imbalance", "imbalance", "15g", "*.csv")
    )
    imnormal_file_names_20g = glob.glob(
        os.path.join(data_dir, "imbalance", "imbalance", "20g", "*.csv")
    )
    imnormal_file_names_25g = glob.glob(
        os.path.join(data_dir, "imbalance", "imbalance", "25g", "*.csv")
    )
    imnormal_file_names_30g = glob.glob(
        os.path.join(data_dir, "imbalance", "imbalance", "30g", "*.csv")
    )

    # concat all file names
    file_names = []
    file_names += normal_file_names
    file_names += imnormal_file_names_6g
    file_names += imnormal_file_names_10g
    file_names += imnormal_file_names_15g
    file_names += imnormal_file_names_20g
    file_names += imnormal_file_names_25g
    file_names += imnormal_file_names_30g
    return file_names


class MachineryIngestor(XIPDataIngestor):
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
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                    rid UInt32,
                    bid UInt32,
                    sensor_0 Float32,
                    sensor_1 Float32,
                    sensor_2 Float32,
                    sensor_3 Float32,
                    sensor_4 Float32,
                    sensor_5 Float32,
                    sensor_6 Float32,
                    sensor_7 Float32,
                    label UInt8,
                    tag String,
                    pid UInt32 -- partition key, used for sampling
                ) ENGINE = MergeTree()
                PARTITION BY pid
                ORDER BY bid
                SETTINGS index_granularity = 32
        """
        self.db_client.command(sql)

    def get_file_ingestion_query(
        self,
        database: str,
        table_name: str,
        table_size: int,
        label: int,
        tag: str,
        start_bid: int,
        file_nrows: int,
        segment_nrows: int,
        nparts: int,
        seed: int,
    ) -> str:
        values = ", ".join([f"sensor_{i} AS sensor_{i}" for i in range(8)])
        values_w_type = ", ".join([f"sensor_{i} Float32" for i in range(8)])
        query = f"""INSERT INTO {database}.{table_name} \
                    SELECT tmp1.rid as rid, tmp1.bid as bid, {values}, tmp1.label as label, \
                            tmp1.tag as tag, tmp2.pid as pid \
                    FROM \
                    ( \
                        SELECT \
                                ({table_size} + rowNumberInAllBlocks()) AS rid, \
                                {label} AS label, {tag} AS tag, \
                                {start_bid} + floor((rowNumberInAllBlocks())/{segment_nrows}) AS bid, \
                                {values} \
                        FROM input('{values_w_type}') \
                    ) as tmp1 \
                    JOIN \
                    ( \
                        SELECT ({table_size} + rowNumberInAllBlocks()) as rid, \
                                random_number % {nparts} as pid \
                        FROM ( \
                            SELECT * \
                            FROM generateRandom('random_number UInt32', {seed}) \
                            LIMIT {file_nrows} \
                        ) \
                    ) as tmp2 \
                    ON tmp1.rid = tmp2.rid \
                    FORMAT CSV \
                """
        return query

    def ingest_data(self) -> None:
        self.logger.info(
            f"Ingesting data from {self.dsrc} into table {self.database}.{self.table}"
        )
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        assert (
            self.dsrc_type == "csv_dir"
        ), f"Unsupported data source type {self.dsrc_type}"

        files = get_raw_data_files_list(self.dsrc)
        file_nrows = 250000
        segments_per_file = 5
        segment_nrows = file_nrows // segments_per_file
        nparts = self.nparts
        for bid, src in tqdm(
            enumerate(files),
            desc=f"Ingesting data to {self.database}.{self.table}",
            total=len(files),
        ):
            filename = os.path.basename(src)
            tag = ".".join(filename.split(".")[:-1])
            dirname = os.path.basename(os.path.dirname(src))
            label = ["normal", "6g", "10g", "15g", "20g", "25g", "30g"].index(dirname)
            cnt = self.db_client.command(
                f"SELECT count(*) FROM {self.database}.{self.table}"
            )
            start_bid = bid * segments_per_file
            query = self.get_file_ingestion_query(
                self.database,
                self.table,
                cnt,
                label,
                tag,
                start_bid,
                file_nrows,
                segment_nrows,
                nparts,
                seed=self.seed,
            )
            command = f"""clickhouse-client --query "{query}" < {src}"""
            os.system(command)

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


class MSensorsIngestor(XIPDataIngestor):
    def __init__(
        self,
        sid: int,
        dsrc_type: str,
        dsrc: str,
        database: str,
        table: str,
        nparts: int,
        seed: int,
    ) -> None:
        super().__init__(dsrc_type, dsrc, database, table, nparts, seed)
        self.sid = sid

    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return

        self.logger.info(f"Creating table {self.database}.{self.table}")
        self.db_client.command(
            f"""CREATE TABLE IF NOT EXISTS {self.database}.{self.table}(
                                rid UInt32,
                                bid UInt32,
                                signal Float32,
                                label UInt8,
                                tag String,
                                pid UInt32 -- partition key, used for sampling
                                ) ENGINE = MergeTree()
                                ORDER BY (pid, bid)
                                SETTINGS index_granularity = 32
                            """
        )

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
        self.db_client.command(
            f"""INSERT INTO {self.database}.{self.table}
                SELECT rid, bid, sensor_{self.sid} as signal, label, tag, pid
                FROM {self.dsrc}"""
        )


def get_ingestor(nparts: int = 100, seed: int = 0):
    possible_dsrcs = [
        "/public/ckchang/db/clickhouse/user_files/machinery",
        "/mnt/sdb/dataset/machinery",
        "/mnt/hddraid/clickhouse-data/user_files/machinery",
        "/var/lib/clickhouse/user_files/machinery"
    ]
    dsrc = None
    for src in possible_dsrcs:
        if os.path.exists(src):
            dsrc = src
            print(f"dsrc path: {dsrc}")
            break
    if dsrc is None:
        raise RuntimeError("no valid dsrc!")

    ingestor = MachineryIngestor(
        dsrc_type="csv_dir",
        dsrc=dsrc,
        database=f"xip_{seed}",
        table=f"mach_imbalance_{nparts}",
        nparts=nparts,
        seed=seed,
    )
    return ingestor


def get_dloader(nparts: int = 100, seed: int = 0, verbose: bool = False) -> XIPDataLoader:
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"mach_imbalance_{nparts}",
        seed=seed,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")
    return data_loader
