from typing import List

from apxinfer.core.utils import XIPFeatureVec, XIPQType, XIPQueryConfig, XIPRequest
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.machinery.data import MachineryRequest


class MachineryQP(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, 
                 data_loader: XIPDataLoader, 
                 dcol: str, 
                 dcol_ops: List[str], 
                 fnames: List[str] = None, 
                 verbose: bool = False) -> None:
        self.dcol = dcol
        self.dcol_ops = dcol_ops
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: MachineryRequest) -> str:
        bid = request["req_bid"]
        qcond = f"bid = {bid}"
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = [self.dcol]
        dcol_aggs = [self.dcol_ops]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: MachineryRequest, qcfg: XIPQueryConfig) -> str:
        return str(request['req_bid'])

    def key_to_request(self, request: MachineryRequest, qcfg: XIPQueryConfig, key: str) -> MachineryRequest:
        new_request = {**request}
        new_request['req_bid'] = int(key)
        return new_request


def get_qps(data_loader: XIPDataLoader, verbose: bool = False, **kwargs):
    nf = kwargs.get("nf", 8)
    qps: List[XIPQueryProcessor] = []
    # dcols = [f'sensor_{i}' for i in range(8)]
    # dcols_aggs = [["avg"] for i in range(8)]
    for i in range(nf):
        qps.append(MachineryQP(qname=f"q-{len(qps)}",
                               qtype=XIPQType.AGG,
                               data_loader=data_loader,
                               dcol=f'sensor_{i}',
                               dcol_ops=["avg"],
                               verbose=verbose))
    return qps


def get_qps_x(data_loader: XIPDataLoader, verbose: bool = False, **kwargs):
    nf = kwargs.get("nf", 8)
    qps: List[XIPQueryProcessor] = []
    # dcols = [f'sensor_{i}' for i in range(8)]
    # dcols_aggs = [["avg"] for i in range(8)]
    for i in range(nf):
        qps.append(MachineryQP(qname=f"q-{len(qps)}",
                               qtype=XIPQType.AGG,
                               data_loader=data_loader,
                               dcol=f'sensor_{i}',
                               dcol_ops=["avg"],
                               verbose=verbose))
    for i in range(nf, 8):
        qps.append(MachineryQP(qname=f"q-{len(qps)}",
                               qtype=XIPQType.KeySearch,
                               data_loader=data_loader,
                               dcol=f'sensor_{i}',
                               dcol_ops=["avg"],
                               verbose=verbose))
    return qps


def get_qps_varynf(data_loader: XIPDataLoader, verbose: bool = False, **kwargs):
    nf = kwargs.get("nf", 8)
    qps: List[XIPQueryProcessor] = []
    # dcols = [f'sensor_{i}' for i in range(8)]
    # dcols_aggs = [["avg"] for i in range(8)]
    for i in range(nf):
        qps.append(MachineryQP(qname=f"q-{len(qps)}",
                               qtype=XIPQType.AGG,
                               data_loader=data_loader,
                               dcol=f'sensor_{i}',
                               dcol_ops=["avg"],
                               verbose=verbose))
    for i in range(nf, 8):
        qps.append(MachineryQP(qname=f"q-{len(qps)}",
                               qtype=XIPQType.ExactAGG,
                               data_loader=data_loader,
                               dcol=f'sensor_{i}',
                               dcol_ops=["avg"],
                               verbose=verbose))
    return qps


class MachinerySimQP(MachineryQP):
    def feature_transformation(self, request: XIPRequest, fvec: XIPFeatureVec) -> XIPFeatureVec:
        fnames = fvec['fnames']
        offsets = [request[f'req_offset_{fname}'] for fname in fnames]
        fvec['fvals'] = fvec['fvals'] + offsets
        return fvec