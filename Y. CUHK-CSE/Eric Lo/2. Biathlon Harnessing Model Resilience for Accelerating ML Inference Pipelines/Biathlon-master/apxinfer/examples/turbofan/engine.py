from typing import List
from apxinfer.core.utils import XIPFeatureVec, XIPQType, XIPRequest
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine

from apxinfer.examples.turbofan.query import TurbofanQPAgg


def get_turbofan_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"turbofan_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    columns = [
        "hs",
        "SmLPC",
        "LPT_flow_mod",
        "HPC_eff_mod",
        "fan_flow_mod",
        "fan_eff_mod",
        "LPT_eff_mod",
        "HPT_flow_mod",
        "HPT_eff_mod",
    ]
    aggops = ["avg"]
    for col in columns:
        qps.append(
            TurbofanQPAgg(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.AGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=aggops,
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_turbofanall_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"turbofan_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    columns = [
        "alt",
        "Mach",
        "TRA",
        "T2",
        "T24",
        "T30",
        "T48",
        "T50",
        "P15",
        "P2",
        "P21",
        "P24",
        "Ps30",
        "P40",
        "P50",
        "Nf",
        "Nc",
        "Wf",
        "T40",
        "P30",
        "P45",
        "W21",
        "W22",
        "W25",
        "W31",
        "W32",
        "W48",
        "W50",
        "SmFan",
        "SmLPC",
        "SmHPC",
        "phi",
        "fan_eff_mod",
        "fan_flow_mod",
        "LPC_eff_mod",
        "LPC_flow_mod",
        "HPC_eff_mod",
        "HPC_flow_mod",
        "HPT_eff_mod",
        "HPT_flow_mod",
        "LPT_eff_mod",
        "LPT_flow_mod",
        # "Y",
        # "unit",
        # "cycle",
        "Fc",
        "hs",
    ]
    aggops = ["avg", "stdPop", "median", "min", "max"]
    for col in columns:
        qps.append(
            TurbofanQPAgg(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.AGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=aggops,
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_turbofan_median_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"turbofan_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    columns = [
        "hs",
        "SmLPC",
        "LPT_flow_mod",
        "HPC_eff_mod",
        "fan_flow_mod",
        "fan_eff_mod",
        "LPT_eff_mod",
        "HPT_flow_mod",
        "HPT_eff_mod",
    ]
    aggops = ["median"]
    for col in columns:
        qps.append(
            TurbofanQPAgg(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.AGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=aggops,
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


class TurbofanQPAggSimMedian(TurbofanQPAgg):
    def feature_transformation(
        self, request: XIPRequest, fvec: XIPFeatureVec
    ) -> XIPFeatureVec:
        return self.feature_transformation_offset(request, fvec)


def get_turbofan_simmedian_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"turbofan_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    columns = [
        "hs",
        "SmLPC",
        "LPT_flow_mod",
        "HPC_eff_mod",
        "fan_flow_mod",
        "fan_eff_mod",
        "LPT_eff_mod",
        "HPT_flow_mod",
        "HPT_eff_mod",
    ]
    aggops = ["median"]
    for col in columns:
        qps.append(
            TurbofanQPAggSimMedian(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.AGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=aggops,
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
