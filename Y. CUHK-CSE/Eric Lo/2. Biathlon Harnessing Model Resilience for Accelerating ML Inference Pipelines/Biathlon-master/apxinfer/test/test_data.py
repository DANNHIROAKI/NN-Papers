import time
import numpy as np
import asyncio
from aiohttp import ClientSession
from aiochclient import ChClient

from apxinfer.core.data import DBHelper

if __name__ == "__main__":

    def get_sql(sql_nthreads: int, start_pid: int, end_pid: int):
        """
        SELECT trip_duration, total_amount, fare_amount
        -- SELECT avg(trip_duration), avg(total_amount), avg(fare_amount)
        FROM xip.trips
        WHERE pid >= 0 AND pid < 100
            AND pickup_datetime >= '2015-08-02 10:00:04'
            AND pickup_datetime < '2015-08-02 11:00:04'
            AND pickup_ntaname = 'Turtle Bay-East Midtown'
            AND dropoff_datetime IS NOT NULL
            AND dropoff_datetime <= '2015-08-02 11:00:04'
        SETTINGS max_threads = 10
        """
        sql = f"""
            SELECT trip_duration, total_amount, fare_amount
            -- SELECT avg(trip_duration), avg(total_amount), avg(fare_amount)
            FROM xip.trips
            WHERE pid >= {start_pid} AND pid < {end_pid}
                AND pickup_datetime >= '2015-08-02 10:00:04'
                AND pickup_datetime < '2015-08-02 11:00:04'
                AND pickup_ntaname = 'Turtle Bay-East Midtown'
                AND dropoff_datetime IS NOT NULL
                AND dropoff_datetime <= '2015-08-02 11:00:04'
            SETTINGS max_threads = {sql_nthreads}"""
        return sql

    ntries = 5
    nparts = 100
    nthreads = 10
    nparts_per_thr = nparts // nthreads
    ncors = 10
    nparts_per_cor = nparts // ncors

    print(f"npart={nparts}, nthreads={nthreads}, ncors={ncors}")
    print(f"nparts_per_thr={nparts_per_thr}")
    print(f"nparts_per_cor={nparts_per_cor}")

    db_client = DBHelper.get_db_client()

    sql = get_sql(1, 0, nparts)
    rrdata = db_client.query_np(sql)
    print(f"rrdata.shape={rrdata.shape}")
    single_ext_loading = 0
    for _ in range(ntries):
        st = time.time()
        sql = get_sql(1, 0, nparts)
        rrdata = db_client.query_np(sql)
        single_ext_loading += time.time() - st

    sql = get_sql(nthreads, 0, nparts)
    rrdata = db_client.query_np(sql)
    print(f"rrdata.shape={rrdata.shape}")
    parallel_ext_loading = 0
    for _ in range(ntries):
        st = time.time()
        sql = get_sql(nthreads, 0, nparts)
        rrdata = db_client.query_np(sql)
        parallel_ext_loading += time.time() - st

    rrdatas = []
    for i in range(0, nparts, nparts_per_thr):
        sql = get_sql(1, i, i + nparts_per_thr)
        rrdatas.append(db_client.query_np(sql))
    rrdata = np.concatenate(rrdatas)
    print(f"rrdata.shape={rrdata.shape}")
    single_apx_loading = 0
    for _ in range(ntries):
        st = time.time()
        rrdatas = []
        for i in range(0, nparts, nparts_per_thr):
            sql = get_sql(1, i, i + nparts_per_thr)
            rrdatas.append(db_client.query_np(sql))
        rrdata = np.concatenate(rrdatas)
        single_apx_loading += time.time() - st

    rrdatas = []
    for i in range(0, nparts, nparts_per_thr):
        sql = get_sql(nthreads, i, i + nparts_per_thr)
        rrdatas.append(db_client.query_np(sql))
    rrdata = np.concatenate(rrdatas)
    print(f"rrdata.shape={rrdata.shape}")
    parallel_apx_loading = 0
    for _ in range(ntries):
        st = time.time()
        rrdatas = []
        for i in range(0, nparts, nparts_per_thr):
            sql = get_sql(nthreads, i, i + nparts_per_thr)
            rrdatas.append(db_client.query_np(sql))
        rrdata = np.concatenate(rrdatas)
        parallel_apx_loading += time.time() - st

    async def asyn_run(ncors: int, sql_nthreads: int):
        client = ChClient(ClientSession(), compress_response=False)
        # making queries in parallel
        results = await asyncio.gather(
            *[
                client.fetch(
                    get_sql(sql_nthreads, i, i + nparts_per_cor),
                    decode=True,
                )
                for i in range(0, nparts_per_thr * ncors, nparts_per_thr)
            ]
        )
        await client.close()
        return np.array([list(row.values()) for result in results for row in result])

    assert nthreads % ncors == 0
    nrounds = nthreads // ncors
    rrdatas = []
    for i in range(nrounds):
        rrdatas.append(asyncio.run(asyn_run(ncors, nrounds)))
    rrdata = np.concatenate(rrdatas)
    print(f"rrdata.shape={rrdata.shape}")

    apx_loading_asynio = 0
    for _ in range(ntries):
        st = time.time()
        assert nthreads % ncors == 0
        nrounds = nthreads // ncors
        rrdatas = []
        for i in range(nrounds):
            rrdatas.append(asyncio.run(asyn_run(ncors, nrounds)))
        rrdata = np.concatenate(rrdatas)
        apx_loading_asynio += time.time() - st

    print(f"single_ext_loading   ={single_ext_loading}")
    print(f"single_apx_loading   ={single_apx_loading}")
    print(f"parallel_ext_loading ={parallel_ext_loading}")
    print(f"parallel_apx_loading ={parallel_apx_loading}")
    print(f"asynio_apx_loading   ={apx_loading_asynio}")
