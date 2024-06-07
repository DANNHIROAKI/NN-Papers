from typing import List
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPQType, XIPQueryConfig
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.trips.data import TripsRequest


class TripsQP0(XIPQueryProcessor):
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        fnames: List[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        self.embeddings = {}
        for dcol in ["pickup_ntaname", "dropoff_ntaname"]:
            self.get_dcol_embeddings(dcol)

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = [
            "req_trip_distance",
            "req_pickup_ntaname",
            "req_dropoff_ntaname",
            "req_pickup_datetime",
        ]
        dcol_aggs = [
            [lambda x: float(x[0][0])],
            [lambda x: self.get_dcol_embeddings("pickup_ntaname")[x[0][1]]],
            [lambda x: self.get_dcol_embeddings("dropoff_ntaname")[x[0][2]]],
            [
                lambda x: int(pd.to_datetime(x[0][3]).weekday() >= 5),
            ],
        ]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class TripsQP1(XIPQueryProcessor):
    def get_query_condition(self, request: TripsRequest) -> str:
        to_dt = pd.to_datetime(request["req_pickup_datetime"])
        from_dt = to_dt - dt.timedelta(hours=1)
        pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
        and_list = [
            f"pickup_datetime >= '{from_dt}'",
            f"pickup_datetime < '{to_dt}'",
            f"pickup_ntaname = '{pickup_ntaname}'",
            # "dropoff_datetime IS NOT NULL",
            # f"dropoff_datetime <= '{to_dt}'",
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["fare_amount"]
        dcol_aggs = [["count", "avg"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TripsRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_pickup_ntaname']

    def key_to_request(self, request: TripsRequest, qcfg: XIPQueryConfig, key: str) -> TripsRequest:
        new_request = {**request}
        new_request['req_pickup_ntaname'] = key
        return new_request


class TripsQP2(XIPQueryProcessor):
    def get_query_condition(self, request: TripsRequest) -> str:
        to_dt = pd.to_datetime(request["req_pickup_datetime"])
        from_dt = to_dt - dt.timedelta(minutes=30)
        # pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
        dropoff_ntaname = request["req_dropoff_ntaname"].replace("'", r"\'")
        and_list = [
            f"dropoff_datetime >= '{from_dt}'",
            f"dropoff_datetime < '{to_dt}'",
            f"dropoff_ntaname = '{dropoff_ntaname}'",
            # "dropoff_datetime IS NOT NULL",
            # f"dropoff_datetime <= '{to_dt}'",
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["fare_amount"]
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TripsRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_dropoff_ntaname']

    def key_to_request(self, request: TripsRequest, qcfg: XIPQueryConfig, key: str) -> TripsRequest:
        new_request = {**request}
        new_request['req_dropoff_ntaname'] = key
        return new_request


class TripsQP3(XIPQueryProcessor):
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        window: float = 1.0,
        fnames: List[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        self.window = window

    def get_query_condition(self, request: TripsRequest) -> str:
        to_dt = pd.to_datetime(request["req_pickup_datetime"])
        from_dt = to_dt - dt.timedelta(hours=self.window)
        pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
        and_list = [
            f"pickup_datetime >= '{from_dt}'",
            f"pickup_datetime < '{to_dt}'",
            f"pickup_ntaname = '{pickup_ntaname}'",
            # "dropoff_datetime IS NOT NULL",
            # f"dropoff_datetime <= '{to_dt}'",
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["fare_amount"]
        dcol_aggs = [["count", "avg"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TripsRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_pickup_ntaname']

    def key_to_request(self, request: TripsRequest, qcfg: XIPQueryConfig, key: str) -> TripsRequest:
        new_request = {**request}
        new_request['req_pickup_ntaname'] = key
        return new_request


class TripsQP4(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType,
                 data_loader: XIPDataLoader, window: float = 0.5,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        self.window = window

    def get_query_condition(self, request: TripsRequest) -> str:
        to_dt = pd.to_datetime(request["req_pickup_datetime"])
        from_dt = to_dt - dt.timedelta(hours=self.window)
        # pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
        dropoff_ntaname = request["req_dropoff_ntaname"].replace("'", r"\'")
        and_list = [
            f"dropoff_datetime >= '{from_dt}'",
            f"dropoff_datetime < '{to_dt}'",
            f"dropoff_ntaname = '{dropoff_ntaname}'",
            # "dropoff_datetime IS NOT NULL",
            # f"dropoff_datetime <= '{to_dt}'",
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["fare_amount"]
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TripsRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_dropoff_ntaname']

    def key_to_request(self, request: TripsRequest, qcfg: XIPQueryConfig, key: str) -> TripsRequest:
        new_request = {**request}
        new_request['req_dropoff_ntaname'] = key
        return new_request


class TripsPickupQP1(XIPQueryProcessor):
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        window: float = 1,
        fnames: List[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        self.window = window

    def get_query_condition(self, request: TripsRequest) -> str:
        to_dt = pd.to_datetime(request["req_pickup_datetime"])
        from_dt = to_dt - dt.timedelta(hours=self.window)
        pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
        and_list = [
            f"pickup_datetime >= '{from_dt}'",
            f"pickup_datetime < '{to_dt}'",
            f"pickup_ntaname = '{pickup_ntaname}'",
            # "dropoff_datetime IS NOT NULL",
            # f"dropoff_datetime < '{to_dt}'",
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["fare_amount"]
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TripsRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_pickup_ntaname']

    def key_to_request(self, request: TripsRequest, qcfg: XIPQueryConfig, key: str) -> TripsRequest:
        new_request = {**request}
        new_request['req_pickup_ntaname'] = key
        return new_request


class TripsPickupQP2(TripsPickupQP1):
    def get_query_condition(self, request: TripsRequest) -> str:
        qcond = super().get_query_condition(request)
        to_dt = pd.to_datetime(request["req_pickup_datetime"])
        and_list = [
            qcond,
            "dropoff_datetime IS NOT NULL",
            f"dropoff_datetime < '{to_dt}'",
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["fare_amount"]
        dcol_aggs = [["avg"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class TripsDropoffQP1(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType,
                 data_loader: XIPDataLoader, window: float = 0.5,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        self.window = window

    def get_query_condition(self, request: TripsRequest) -> str:
        to_dt = pd.to_datetime(request["req_pickup_datetime"])
        from_dt = to_dt - dt.timedelta(hours=self.window)
        dropoff_ntaname = request["req_dropoff_ntaname"].replace("'", r"\'")
        and_list = [
            f"dropoff_datetime >= '{from_dt}'",
            f"dropoff_datetime < '{to_dt}'",
            f"dropoff_ntaname = '{dropoff_ntaname}'",
            "dropoff_datetime IS NOT NULL",
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["fare_amount"]
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: TripsRequest, qcfg: XIPQueryConfig) -> str:
        return request['req_dropoff_ntaname']

    def key_to_request(self, request: TripsRequest, qcfg: XIPQueryConfig, key: str) -> TripsRequest:
        new_request = {**request}
        new_request['req_dropoff_ntaname'] = key
        return new_request
