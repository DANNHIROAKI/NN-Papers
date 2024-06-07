from typing import List
from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine


from apxinfer.examples.machinery.data import MachineryRequest, get_dloader
from apxinfer.examples.machinery.query import MachineryQP, MachinerySimQP

def get_qengine(qps: List[XIPQueryProcessor], ncores: int = 0, verbose: bool = False):
    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_machineryralfmedian_engine(nparts: int = 100, ncores: int = 0,
                               seed: int = 0, median_qids: List[int] = [0],
                               verbose: bool = False) -> XIPFEngine:
    data_loader: XIPDataLoader = get_dloader(nparts, seed, verbose)
    qps: List[XIPQueryProcessor] = []
    nf = 8
    for i in range(nf):
        if i in median_qids:
            dcol_ops=["median"]
        else:
            dcol_ops=["avg"]
        qps.append(MachineryQP(qname=f"q-{len(qps)}",
                               qtype=XIPQType.AGG,
                               data_loader=data_loader,
                               dcol=f'sensor_{i}',
                               dcol_ops=dcol_ops,
                               verbose=verbose))
    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_machineryralfsimmedian_engine(nparts: int = 100, ncores: int = 0,
                               seed: int = 0, median_qids: List[int] = [0],
                               verbose: bool = False) -> XIPFEngine:
    data_loader: XIPDataLoader = get_dloader(nparts, seed, verbose)
    qps: List[XIPQueryProcessor] = []
    nf = 8
    for i in range(nf):
        if i in median_qids:
            dcol_ops=["median"]
            qps.append(MachinerySimQP(qname=f"q-{len(qps)}",
                                qtype=XIPQType.AGG,
                                data_loader=data_loader,
                                dcol=f'sensor_{i}',
                                dcol_ops=dcol_ops,
                                verbose=verbose))
        else:
            dcol_ops=["avg"]
            qps.append(MachineryQP(qname=f"q-{len(qps)}",
                                qtype=XIPQType.AGG,
                                data_loader=data_loader,
                                dcol=f'sensor_{i}',
                                dcol_ops=dcol_ops,
                                verbose=verbose))
    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
