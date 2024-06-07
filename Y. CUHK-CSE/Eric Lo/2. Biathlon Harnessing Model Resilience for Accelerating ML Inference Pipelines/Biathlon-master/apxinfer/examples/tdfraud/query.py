from typing import List
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPQType, XIPQueryConfig
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.tdfraud.data import TDFraudRequest


class TDFraudQP0(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["req_app", "req_device", "req_os",
                 "req_channel", "req_click_time"]
        dcol_aggs = [
            [lambda x: int(x[0][0])],
            [lambda x: int(x[0][1])],
            [lambda x: int(x[0][2])],
            [lambda x: int(x[0][3])],
            [lambda x: pd.to_datetime(x[0][4]).hour,
             lambda x: pd.to_datetime(x[0][4]).day],
        ]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class TDFraudQP1(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TDFraudRequest) -> str:
        window_size = 1
        req_dt = pd.to_datetime(request["req_click_time"])
        from_dt = req_dt + dt.timedelta(days=-window_size)
        ip = request["req_ip"]
        and_list = [f"ip = {ip}",
                    f"click_time >= '{from_dt}'",
                    f"click_time < '{req_dt}'"]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ['channel']
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TDFraudRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_ip']

    def key_to_request(self, request: TDFraudRequest, qcfg: XIPQueryConfig, key: str) -> TDFraudRequest:
        new_request = {**request}
        new_request['req_ip'] = key
        return new_request


class TDFraudQP2(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TDFraudRequest) -> str:
        req_dt = pd.to_datetime(request["req_click_time"])
        ip = request["req_ip"]
        app = request["req_app"]
        and_list = [f"ip = {ip}",
                    f"app = {app}",
                    f"click_time < '{req_dt}'"]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ['channel']
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TDFraudRequest, qcfg: XIPQueryConfig) -> str:
        ip = request["req_ip"]
        app = request["req_app"]
        return f"{ip}_{app}"

    def key_to_request(self, request: TDFraudRequest, qcfg: XIPQueryConfig, key: str) -> TDFraudRequest:
        new_request = {**request}
        ip, app = key.split("_")
        new_request['req_ip'] = ip
        new_request['req_app'] = app
        return new_request


class TDFraudQP3(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TDFraudRequest) -> str:
        req_dt = pd.to_datetime(request["req_click_time"])
        ip = request["req_ip"]
        app = request["req_app"]
        os = request["req_os"]
        and_list = [f"ip = {ip}",
                    f"app = {app}",
                    f"os = {os}",
                    f"click_time < '{req_dt}'"]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ['channel']
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TDFraudRequest, qcfg: XIPQueryConfig) -> str:
        ip = request["req_ip"]
        app = request["req_app"]
        os = request["req_os"]
        return f"{ip}_{app}_{os}"

    def key_to_request(self, request: TDFraudRequest, qcfg: XIPQueryConfig, key: str) -> TDFraudRequest:
        new_request = {**request}
        ip, app, os = key.split("_")
        new_request['req_ip'] = ip
        new_request['req_app'] = app
        new_request['req_os'] = os
        return new_request


class TDFraudKaggleQP0(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["req_app", "req_device", "req_os",
                 "req_channel", "req_click_time"]
        dcol_aggs = [
            [lambda x: int(x[0][0])],
            [lambda x: int(x[0][1])],
            [lambda x: int(x[0][2])],
            [lambda x: int(x[0][3])],
            [lambda x: pd.to_datetime(x[0][4]).day_of_week,
             lambda x: pd.to_datetime(x[0][4]).day_of_year],
        ]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class TDFraudKaggleQP1(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TDFraudRequest) -> str:
        req_dt = pd.to_datetime(request["req_click_time"])
        ip = request["req_ip"]
        and_list = [f"ip = {ip}",
                    f"click_time < '{req_dt}'"]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ['channel']
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TDFraudRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_ip']

    def key_to_request(self, request: TDFraudRequest, qcfg: XIPQueryConfig, key: str) -> TDFraudRequest:
        new_request = {**request}
        new_request['req_ip'] = key
        return new_request