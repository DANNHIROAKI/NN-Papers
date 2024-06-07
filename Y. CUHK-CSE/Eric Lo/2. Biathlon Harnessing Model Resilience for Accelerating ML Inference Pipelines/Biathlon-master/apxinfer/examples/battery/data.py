import os
from tqdm import tqdm
import pandas as pd

from apxinfer.core.utils import XIPRequest
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


class BatteryRequest(XIPRequest):
    req_bid: int
    req_time: float


class BatteryIngestor(XIPDataIngestor):
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
        # Voltage_measured,Current_measured,Temperature_measured,Current_load,Voltage_load,Time
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                    rid UInt32, -- row id
                    bid UInt32, -- block id, also file id
                    Voltage_measured Float32,
                    Current_measured Float32,
                    Temperature_measured Float32,
                    Current_load Float32,
                    Voltage_load Float32,
                    Time Float32,
                    Timestamp DateTime64,
                    pid UInt32, -- partition key, used for sampling
                    INDEX idx1 bid TYPE set(0) GRANULARITY 1,
                ) ENGINE = MergeTree()
                PARTITION BY pid
                ORDER BY Timestamp
                -- SETTINGS index_granularity = 32
        """
        self.db_client.command(sql)

    def get_file_ingestion_query(
        self,
        database: str,
        table_name: str,
        table_size: int,
        bid: int,
        start_datetime: str,
        file_nrows: int,
        nparts: int,
        seed: int,
    ) -> str:
        col_names = ["Voltage_measured", "Current_measured", "Temperature_measured", "Current_load", "Voltage_load", "Time"]
        values = ", ".join([f"{cname} AS {cname}" for cname in col_names])
        values_w_type = ", ".join([f"{cname} Float32" for cname in col_names])
        query = f"""INSERT INTO {database}.{table_name} \
                    SELECT tmp1.rid as rid, {bid} as bid, {values}, \
                            tmp1.Timestamp as Timestamp, tmp2.pid as pid \
                    FROM \
                    ( \
                        SELECT \
                                ({table_size} + rowNumberInAllBlocks()) AS rid, \
                                addMilliseconds(parseDateTime64BestEffort('{start_datetime}'), Time*1000) AS Timestamp, \
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
                    FORMAT CSVWithNames \
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
        meta_data = pd.read_csv(os.path.join(self.dsrc, "metadata.csv"))
        # selected_type = 'discharge'
        selected_type = 'charge'
        selected_data = meta_data[meta_data['type'] == selected_type]
        # iterate row as dict
        for row in tqdm(selected_data.to_dict(orient='records')):
            filename = row['filename']
            start_time_list = row['start_time'].strip("[]").split()
            start_time_list = [float(e) for e in start_time_list]
            assert len(start_time_list) == 6
            start_time = f"{int(start_time_list[0])}-{int(start_time_list[1])}-{int(start_time_list[2])} {int(start_time_list[3])}:{int(start_time_list[4])}:{start_time_list[5]}"
            bid = row['uid']
            src = os.path.join(self.dsrc, 'data', filename)
            max_file_nrows = 4000

            cnt = self.db_client.command(
                f"SELECT count(*) FROM {self.database}.{self.table}"
            )

            query = self.get_file_ingestion_query(
                self.database,
                self.table,
                cnt,
                bid,
                start_time,
                max_file_nrows,
                self.nparts,
                self.seed,
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


def get_dsrc():
    possible_dsrcs = [
        "/public/ckchang/db/clickhouse/user_files/nasa-battery/cleaned_dataset",
        "/mnt/sdb/dataset/nasa-battery/cleaned_dataset",
        "/mnt/hddraid/clickhouse-data/user_files/nasa-battery/cleaned_dataset",
        "/var/lib/clickhouse/user_files/nasa-battery/cleaned_dataset"
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


def get_ingestor(nparts: int = 100, seed: int = 0):
    dsrc = get_dsrc()

    ingestor = BatteryIngestor(
        dsrc_type="csv_dir",
        dsrc=dsrc,
        database=f"xip_{seed}",
        table=f"battery_{nparts}",
        nparts=nparts,
        seed=seed,
    )
    return ingestor


def get_dloader(nparts: int = 100, seed: int = 0, verbose: bool = False) -> XIPDataLoader:
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"battery_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")
    return data_loader


def ingest(nparts: int = 100, seed: int = 0, verbose: bool = False):
    ingestor = get_ingestor(nparts=nparts, seed=seed)
    ingestor.run()


if __name__ == "__main__":
    ingestor = get_ingestor(nparts=100, seed=0)
    ingestor.run()
