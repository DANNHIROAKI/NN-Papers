from typing import List

from apxinfer.core.utils import XIPQType, XIPQueryConfig
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.battery.data import BatteryRequest


class BatteryQPNonAGG(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["req_time"]
        dcol_aggs = [
            [lambda x: int(x[0][0])]
        ]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class BatteryQPAgg(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType,
                 data_loader: XIPDataLoader,
                 dcol: str,
                 dcol_ops: List[str],
                 fnames: List[str] = None,
                 verbose: bool = False) -> None:
        self.dcol = dcol
        self.dcol_ops = dcol_ops
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: BatteryRequest) -> str:
        bid = request["req_bid"]
        req_time = request["req_time"]
        qcond = f"bid = {bid} AND Time < {req_time}"
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = [self.dcol]
        dcol_aggs = [self.dcol_ops]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: BatteryRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_bid']

    def key_to_request(self, request: BatteryRequest, qcfg: XIPQueryConfig, key: str) -> BatteryRequest:
        new_request = {**request}
        new_request['req_bid'] = key
        return new_request
