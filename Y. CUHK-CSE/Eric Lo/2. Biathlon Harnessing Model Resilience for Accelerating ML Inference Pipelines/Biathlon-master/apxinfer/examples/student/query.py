from typing import List
import pandas as pd
import numpy as np

from apxinfer.core.utils import XIPRequest, XIPQType, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec
from apxinfer.core.utils import is_same_float
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.student.data import StudentRequest


def get_query_group(q_no: int) -> str:
    # Select level group for the question based on the q_no.
    if q_no <= 3:
        grp = "0-4"
    elif q_no <= 13:
        grp = "5-12"
    elif q_no <= 22:
        grp = "13-22"
    return grp


class StudentQP0(XIPQueryProcessor):
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        fnames: List[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["req_qno"]
        dcol_aggs = [[lambda x: int(x[0][0])]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class StudentQP1(XIPQueryProcessor):
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        dcol: str,
        dcol_ops: List[str], 
        fnames: List[str] = None,
        verbose: bool = False,
    ) -> None:
        self.dcol = dcol
        self.dcol_ops = dcol_ops
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: StudentRequest) -> str:
        req_session_id = request["req_session_id"]
        req_qno = request["req_qno"]
        level_group = get_query_group(req_qno)

        and_list = [
            f"session_id = {req_session_id}",
            f"level_group = '{level_group}'",
            f"isNotNull({self.dcol})"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = [self.dcol]
        dcol_aggs = [self.dcol_ops]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops

    def request_to_key(self, request: StudentRequest, qcfg: XIPQueryConfig) -> str:
        session_id = request['req_session_id']
        req_qno = request["req_qno"]
        # level_group = get_query_group(req_qno)
        return f"{session_id}_{req_qno}"

    def key_to_request(self, request: StudentRequest, qcfg: XIPQueryConfig, key: str) -> StudentRequest:
        new_request = {**request}
        session_id, req_qno = key.split("_")
        new_request["req_session_id"] = session_id
        new_request["req_qno"] = int(req_qno)
        return new_request

    # def get_default_fvec(
    #     self, request: XIPRequest, qcfg: XIPQueryConfig
    # ) -> XIPFeatureVec:
    #     fvals = -np.ones(len(self.fnames))
    #     if is_same_float(qcfg["qsample"], 1.0):
    #         fests = np.zeros(len(self.fnames))
    #     else:
    #         fests = np.ones(len(self.fnames)) * 1e9
    #     return XIPFeatureVec(
    #         fnames=self.fnames,
    #         fvals=fvals,
    #         fests=fests,
    #         fdists=["normal"] * len(self.fnames),
    #     )
