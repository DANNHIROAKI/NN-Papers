from typing import List
import pandas as pd
import datetime as dt

from apxinfer.core.data import DBHelper

from apxinfer.core.utils import XIPQType, XIPQueryConfig
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.tick.data import TickRequest


def get_embedding() -> dict:
    db_client = DBHelper.get_db_client()
    cpairs: pd.DataFrame = db_client.query_df(
        "select distinct cpair from xip.tick_fstore_hour order by cpair"
    )
    currencies = []
    for cpair in cpairs["cpair"].values:
        currencies.append(cpair.split("/")[0])
        currencies.append(cpair.split("/")[1])
    currencies = sorted(list(set(currencies)))
    currency_map = {currency: i + 1 for i, currency in enumerate(currencies)}
    return currency_map


class TickQP0(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        self.embeddings = get_embedding()

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["req_cpair"]
        dcol_aggs = [[lambda x: self.embeddings.get(x[0][0].split("/")[0], 0),
                      lambda x: self.embeddings.get(x[0][0].split("/")[1], 0)]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TickRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_cpair']

    def key_to_request(self, request: TickRequest, qcfg: XIPQueryConfig, key: str) -> TickRequest:
        new_request = {**request}
        new_request['req_cpair'] = key
        return new_request


class TickQP1(XIPQueryProcessor):
    def get_query_condition(self, request: TickRequest) -> str:
        cpair = request["req_cpair"]
        tick_dt = pd.to_datetime(request["req_dt"])
        from_dt = tick_dt
        to_dt = from_dt + dt.timedelta(hours=1)
        and_list = [
            f"cpair = '{cpair}'",
            f"tick_dt >= '{from_dt}'",
            f"tick_dt < '{to_dt}'"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["bid"]
        dcol_aggs = [["avg"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TickRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_cpair']

    def key_to_request(self, request: TickRequest, qcfg: XIPQueryConfig, key: str) -> TickRequest:
        new_request = {**request}
        new_request['req_cpair'] = key
        return new_request


class TickQP2(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 offset: int = 1,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        self.offset = offset
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TickRequest) -> str:
        cpair = request["req_cpair"]
        tick_dt = pd.to_datetime(request["req_dt"])
        target_dt = pd.to_datetime(tick_dt) - dt.timedelta(hours=self.offset)
        and_list = [
            f"cpair = '{cpair}'",
            f"year = {target_dt.year}",
            f"month = {target_dt.month}",
            f"day = {target_dt.day}",
            f"hour = {target_dt.hour}"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["avg_bid"]
        dcol_aggs = [[lambda x: x[0][0]]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class TickQP3(XIPQueryProcessor):
    def get_query_condition(self, request: TickRequest) -> str:
        cpair = request["req_cpair"]
        tick_dt = pd.to_datetime(request["req_dt"])
        from_dt = tick_dt
        to_dt = from_dt + dt.timedelta(hours=1)
        and_list = [
            f"cpair = '{cpair}'",
            f"tick_dt >= '{from_dt}'",
            f"tick_dt < '{to_dt}'"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["ask"]
        dcol_aggs = [["avg"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TickRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_cpair']

    def key_to_request(self, request: TickRequest, qcfg: XIPQueryConfig, key: str) -> TickRequest:
        new_request = {**request}
        new_request['req_cpair'] = key
        return new_request


class TickQP4(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 offset: int = 1,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        self.offset = offset
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TickRequest) -> str:
        cpair = request["req_cpair"]
        tick_dt = pd.to_datetime(request["req_dt"])
        target_dt = pd.to_datetime(tick_dt) - dt.timedelta(hours=self.offset)
        and_list = [
            f"cpair = '{cpair}'",
            f"year = {target_dt.year}",
            f"month = {target_dt.month}",
            f"day = {target_dt.day}",
            f"hour = {target_dt.hour}"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["avg_ask"]
        dcol_aggs = [[lambda x: x[0][0]]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops
