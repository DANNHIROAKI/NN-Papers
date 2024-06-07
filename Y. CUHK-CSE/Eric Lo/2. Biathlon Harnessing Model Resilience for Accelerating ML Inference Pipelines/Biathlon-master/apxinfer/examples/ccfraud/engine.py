from typing import List
from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine

from apxinfer.examples.ccfraud.query import CCFraudQP0
from apxinfer.examples.ccfraud.query import CCFraudQP1
from apxinfer.examples.ccfraud.query import CCFraudQP2
from apxinfer.examples.ccfraud.query import CCFraudQP3
from apxinfer.examples.ccfraud.query import CCFraudQP4
from apxinfer.examples.ccfraud.query import CCFraudQP5
from apxinfer.examples.ccfraud.query import CCFraudQP6


def get_ccfraud_engine(nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False):
    txns_data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"ccfraud_txns_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={txns_data_loader.statistics['tsize']}")
        print(f"nparts={txns_data_loader.statistics['nparts']}")

    cards_data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table="ccfraud_cards",
        seed=0,
        enable_cache=False,
    )

    users_data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table="ccfraud_users",
        seed=0,
        enable_cache=False,
    )

    qp0 = CCFraudQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=txns_data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = CCFraudQP1(
        qname="q1",
        qtype=XIPQType.FSTORE,
        data_loader=cards_data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp2 = CCFraudQP2(
        qname="q2",
        qtype=XIPQType.FSTORE,
        data_loader=users_data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp3 = CCFraudQP3(
        qname="q3",
        qtype=XIPQType.AGG,
        data_loader=txns_data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp4 = CCFraudQP4(
        qname="q4",
        qtype=XIPQType.AGG,
        data_loader=txns_data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp5 = CCFraudQP5(
        qname="q5",
        qtype=XIPQType.AGG,
        data_loader=txns_data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp6 = CCFraudQP6(
        qname="q6",
        qtype=XIPQType.AGG,
        data_loader=txns_data_loader,
        fnames=None,
        verbose=verbose,
    )

    # qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3, qp4]
    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3, qp4, qp5, qp6]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
