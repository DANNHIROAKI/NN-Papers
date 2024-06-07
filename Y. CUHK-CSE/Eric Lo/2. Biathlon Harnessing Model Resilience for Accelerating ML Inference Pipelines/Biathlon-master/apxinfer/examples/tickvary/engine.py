from typing import List
from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine

from apxinfer.examples.tick.query import TickQP1, TickQP2


def get_tick_engine(nparts: int, ncores: int = 0,
                    seed: int = 0, num_months: int = 1,
                    verbose: bool = False):
    tick_data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"tickvary_{num_months}_{nparts}",
        seed=0,
        enable_cache=False,
    )

    tick_fstore_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table="tick_fstore_hour",
        seed=0,
        enable_cache=False,
    )

    qps: List[XIPQueryProcessor] = []
    for offset in range(1, 7):
        qps.append(TickQP2(
            qname=f"q{len(qps)}",
            qtype=XIPQType.FSTORE,
            data_loader=tick_fstore_loader,
            offset=7 - offset,
            fnames=None,
            verbose=verbose,
        ))
    qps.append(TickQP1(
        qname=f"q{len(qps)}",
        qtype=XIPQType.AGG,
        data_loader=tick_data_loader,
        fnames=None,
        verbose=verbose,
    ))
    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
