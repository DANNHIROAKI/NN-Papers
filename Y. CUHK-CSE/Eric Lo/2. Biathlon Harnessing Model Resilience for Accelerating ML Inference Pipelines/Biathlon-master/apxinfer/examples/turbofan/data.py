import os
from tqdm import tqdm
import pandas as pd

from apxinfer.core.utils import XIPRequest
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


class TurbofanRequest(XIPRequest):
    req_name: str
    req_unit: int
    req_cycle: int


class TurbofanIngestor(XIPDataIngestor):
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
        cols = [
            "alt",
            "Mach",
            "TRA",
            "T2",
            "T24",
            "T30",
            "T48",
            "T50",
            "P15",
            "P2",
            "P21",
            "P24",
            "Ps30",
            "P40",
            "P50",
            "Nf",
            "Nc",
            "Wf",
            "T40",
            "P30",
            "P45",
            "W21",
            "W22",
            "W25",
            "W31",
            "W32",
            "W48",
            "W50",
            "SmFan",
            "SmLPC",
            "SmHPC",
            "phi",
            "fan_eff_mod",
            "fan_flow_mod",
            "LPC_eff_mod",
            "LPC_flow_mod",
            "HPC_eff_mod",
            "HPC_flow_mod",
            "HPT_eff_mod",
            "HPT_flow_mod",
            "LPT_eff_mod",
            "LPT_flow_mod",
            "Y",
            "unit",
            "cycle",
            "Fc",
            "hs",
        ]
        int_cols = ["Y", "unit", "cycle", "Fc", "hs"]
        col_with_type = [
            f"{col} UInt32" if col in int_cols else f"{col} Float32" for col in cols
        ]
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                    rid UInt32, -- row id
                    name String,
                    {', '.join(col_with_type)},
                    pid UInt32 -- partition key, used for sampling
                    # INDEX idx1 name TYPE set(0) GRANULARITY 1,
                    # INDEX idx2 unit TYPE set(0) GRANULARITY 1,
                    # INDEX idx3 cycle TYPE set(0) GRANULARITY 1,
                ) ENGINE = MergeTree()
                PARTITION BY pid
                ORDER BY (unit, cycle)
        """
        self.db_client.command(sql)

    def get_file_ingestion_query(
        self,
        database: str,
        table_name: str,
        table_size: int,
        name: str,
        file_nrows: int,
        nparts: int,
        seed: int,
    ) -> str:
        cols = [
            "alt",
            "Mach",
            "TRA",
            "T2",
            "T24",
            "T30",
            "T48",
            "T50",
            "P15",
            "P2",
            "P21",
            "P24",
            "Ps30",
            "P40",
            "P50",
            "Nf",
            "Nc",
            "Wf",
            "T40",
            "P30",
            "P45",
            "W21",
            "W22",
            "W25",
            "W31",
            "W32",
            "W48",
            "W50",
            "SmFan",
            "SmLPC",
            "SmHPC",
            "phi",
            "fan_eff_mod",
            "fan_flow_mod",
            "LPC_eff_mod",
            "LPC_flow_mod",
            "HPC_eff_mod",
            "HPC_flow_mod",
            "HPT_eff_mod",
            "HPT_flow_mod",
            "LPT_eff_mod",
            "LPT_flow_mod",
            "Y",
            "unit",
            "cycle",
            "Fc",
            "hs",
        ]
        int_cols = ["Y", "unit", "cycle", "Fc", "hs"]

        # values = ", ".join([f"{cname} AS {cname}" for cname in cols])
        # col_with_type = [f"{col} UInt32" if col in int_cols else f"{col} Float32" for col in cols]
        # values_w_type = ", ".join(col_with_type)

        values = ", ".join(
            [
                f"toUInt32({cname}) AS {cname}"
                if cname in int_cols
                else f"{cname} AS {cname}"
                for cname in cols
            ]
        )
        values_w_type = ", ".join([f"{cname} Float32" for cname in cols])
        query = f"""INSERT INTO {database}.{table_name} \
                    SELECT tmp1.rid as rid, '{name}' as name, {values}, \
                            tmp2.pid as pid \
                    FROM \
                    ( \
                        SELECT \
                                ({table_size} + rowNumberInAllBlocks()) AS rid, \
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
        train_raw_folder = os.path.join(self.dsrc, "train", "train")
        for filename in tqdm(os.listdir(train_raw_folder)):
            name = filename.split(".")[0]
            src = os.path.join(train_raw_folder, filename)
            df = pd.read_csv(src)
            file_nrows = len(df)

            cnt = self.db_client.command(
                f"SELECT count(*) FROM {self.database}.{self.table}"
            )

            query = self.get_file_ingestion_query(
                self.database,
                self.table,
                cnt,
                name,
                file_nrows,
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
        "/public/ckchang/db/clickhouse/user_files/turbofan",
        "/mnt/sdb/dataset/turbofan",
        "/mnt/hddraid/clickhouse-data/user_files/turbofan",
        "/var/lib/clickhouse/user_files/turbofan",
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

    ingestor = TurbofanIngestor(
        dsrc_type="csv_dir",
        dsrc=dsrc,
        database=f"xip_{seed}",
        table=f"turbofan_{nparts}",
        nparts=nparts,
        seed=seed,
    )
    return ingestor


def get_dloader(nparts: int = 100, seed: int = 0,
                verbose: bool = False) -> XIPDataLoader:
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"turbofan_{nparts}",
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
