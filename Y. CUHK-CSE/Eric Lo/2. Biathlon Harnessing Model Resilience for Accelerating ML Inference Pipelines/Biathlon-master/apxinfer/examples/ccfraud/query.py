from typing import List
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.ccfraud.data import CCFraudRequest


class CCFraudQP0(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        cat_fnames = [
            "use_chip",
            "merchant_name",
            "merchant_city",
            "merchant_state",
        ]
        self.embeddings = {}
        for dcol in cat_fnames:
            self.get_dcol_embeddings(dcol)

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["req_txn_datetime",
                 "req_amount", "req_zip_code", "req_mcc",
                 "req_use_chip", "req_merchant_name",
                 "req_merchant_city", "req_merchant_state"]
        dcol_aggs = [
            [lambda x: pd.to_datetime(x[0][0]).hour],
            [lambda x: float(x[0][1])],
            [lambda x: int(x[0][2])],
            [lambda x: int(x[0][3])],
            [lambda x: self.get_dcol_embeddings("use_chip").get(x[0][4], 0)],
            [lambda x: self.get_dcol_embeddings("merchant_name").get(x[0][5], 0)],
            [lambda x: self.get_dcol_embeddings("merchant_city").get(x[0][6], 0)],
            [lambda x: self.get_dcol_embeddings("merchant_state").get(x[0][7], 0)],
        ]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class CCFraudQP1(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        self.cat_fnames = ["card_brand", "card_type"]
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        self.embeddings = {}
        for dcol in self.cat_fnames:
            self.get_dcol_embeddings(dcol)

    def get_query_condition(self, request: CCFraudRequest) -> str:
        uid = request["req_uid"]
        card_index = request["req_card_index"]
        and_list = [
            f"uid = {uid}",
            f"card_index = {card_index}"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = self.cat_fnames
        dcol_aggs = [[lambda x: self.get_dcol_embeddings(self.cat_fnames[i]).get(x[0][i], 0)]
                      for i in range(len(self.cat_fnames))]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class CCFraudQP2(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        self.num_fnames = [
            "current_age",
            "retirement_age",
            "birth_year",
            "birth_month",
            "gender",
            "per_capita_income",
            "yearly_income",
            "total_debt",
            "fico_score",
            "num_credit_cards",
        ]
        self.cat_fnames = ["uname", "address",
                           "apartment", "city",
                           "state", "zipcode"]
        super().__init__(qname, qtype, data_loader, fnames, verbose)
        self.embeddings = {}
        for dcol in self.cat_fnames:
            self.get_dcol_embeddings(dcol)

    def get_query_condition(self, request: CCFraudRequest) -> str:
        uid = request["req_uid"]
        qcond = f"uid = {uid}"
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = self.num_fnames + self.cat_fnames
        dcol_aggs = [[lambda x: int(x[0][i])]
                     for i in range(len(self.num_fnames))]
        dcol_aggs += [[lambda x: self.get_dcol_embeddings(self.cat_fnames[i]).get(x[0][i + len(self.num_fnames)], 0)]
                      for i in range(len(self.cat_fnames))]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class CCFraudQP3(XIPQueryProcessor):
    def get_query_condition(self, request: CCFraudRequest) -> str:
        window_size = 30
        req_dt = pd.to_datetime(request["req_txn_datetime"])
        from_dt = req_dt + dt.timedelta(days=-window_size)
        req_uid = request["req_uid"]
        and_list = [
            f"txn_datetime >= '{from_dt}'",
            f"txn_datetime < '{req_dt}'",
            f"uid = '{req_uid}'"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["is_fraud"]
        dcol_aggs = [["sum"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class CCFraudQP4(XIPQueryProcessor):
    def get_query_condition(self, request: CCFraudRequest) -> str:
        window_size = 30 * 12
        req_dt = pd.to_datetime(request["req_txn_datetime"])
        from_dt = req_dt + dt.timedelta(days=-window_size)
        req_uid = request["req_uid"]
        req_card = request['req_card_index']
        and_list = [
            f"txn_datetime >= '{from_dt}'",
            f"txn_datetime < '{req_dt}'",
            f"uid = '{req_uid}'",
            f"card_index = '{req_card}'"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["is_fraud"]
        dcol_aggs = [["sum"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class CCFraudQP5(XIPQueryProcessor):
    def get_query_condition(self, request: CCFraudRequest) -> str:
        window_size = 30
        req_dt = pd.to_datetime(request["req_txn_datetime"])
        from_dt = req_dt + dt.timedelta(days=-window_size)
        req_uid = request["req_uid"]
        and_list = [
            f"txn_datetime >= '{from_dt}'",
            f"txn_datetime < '{req_dt}'",
            f"uid = '{req_uid}'"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["amount"]
        dcol_aggs = [["avg"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class CCFraudQP6(XIPQueryProcessor):
    def get_query_condition(self, request: CCFraudRequest) -> str:
        window_size = 30 * 12
        req_dt = pd.to_datetime(request["req_txn_datetime"])
        from_dt = req_dt + dt.timedelta(days=-window_size)
        req_uid = request["req_uid"]
        req_card = request['req_card_index']
        and_list = [
            f"txn_datetime >= '{from_dt}'",
            f"txn_datetime < '{req_dt}'",
            f"uid = '{req_uid}'",
            f"card_index = '{req_card}'"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["amount"]
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops