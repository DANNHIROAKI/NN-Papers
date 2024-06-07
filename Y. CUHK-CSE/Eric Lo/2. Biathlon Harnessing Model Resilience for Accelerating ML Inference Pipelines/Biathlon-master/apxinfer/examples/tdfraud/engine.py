from typing import List
from apxinfer.core.utils import XIPFeatureVec, XIPQType, XIPRequest
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription
from apxinfer.core.fengine import XIPFEngine

from apxinfer.examples.tdfraud.query import TDFraudQP0
from apxinfer.examples.tdfraud.query import TDFraudQP1
from apxinfer.examples.tdfraud.query import TDFraudQP2
from apxinfer.examples.tdfraud.query import TDFraudQP3
from apxinfer.examples.tdfraud.query import TDFraudKaggleQP0
from apxinfer.examples.tdfraud.query import TDFraudKaggleQP1


def get_tdfraud_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"tdfraud_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qp0 = TDFraudQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TDFraudQP1(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp2 = TDFraudQP2(
        qname="q2",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp3 = TDFraudQP3(
        qname="q3",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )

    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_tdfraudkaggle_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"tdfraud_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qp0 = TDFraudKaggleQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TDFraudKaggleQP1(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qps: List[XIPQueryProcessor] = [qp0, qp1]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


class TDFraudQP3Median(TDFraudQP3):
    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["channel"]
        dcol_aggs = [["median"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


def get_tdfraud_median_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"tdfraud_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qp0 = TDFraudQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TDFraudQP1(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp2 = TDFraudQP2(
        qname="q2",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp3 = TDFraudQP3Median(
        qname="q3",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )

    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


class TDFraudQP1Median(TDFraudQP1):
    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["channel"]
        dcol_aggs = [["median"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


def get_tdfraud_medianv2_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"tdfraud_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qp0 = TDFraudQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TDFraudQP1Median(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp2 = TDFraudQP2(
        qname="q2",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp3 = TDFraudQP3(
        qname="q3",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )

    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


class TDFraudQP1SimMedian(TDFraudQP1Median):
    def feature_transformation(
        self, request: XIPRequest, fvec: XIPFeatureVec
    ) -> XIPFeatureVec:
        return self.feature_transformation_offset(request, fvec)


def get_tdfraud_simmedianv2_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"tdfraud_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qp0 = TDFraudQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TDFraudQP1SimMedian(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp2 = TDFraudQP2(
        qname="q2",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp3 = TDFraudQP3(
        qname="q3",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )

    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


class TDFraudQP3SimMedian(TDFraudQP3Median):
    def feature_transformation(
        self, request: XIPRequest, fvec: XIPFeatureVec
    ) -> XIPFeatureVec:
        return self.feature_transformation_offset(request, fvec)


def get_tdfraud_simmedian_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"tdfraud_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qp0 = TDFraudQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TDFraudQP1(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp2 = TDFraudQP2(
        qname="q2",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp3 = TDFraudQP3SimMedian(
        qname="q3",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )

    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
