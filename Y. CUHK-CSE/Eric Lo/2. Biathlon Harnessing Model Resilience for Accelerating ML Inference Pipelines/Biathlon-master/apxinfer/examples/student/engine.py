from typing import List
from apxinfer.core.utils import XIPFeatureVec, XIPQType, XIPRequest
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine

from apxinfer.examples.student.query import StudentQP0
from apxinfer.examples.student.query import StudentQP1


STUDENT_CATEGORICAL = ["event_name", "name", "fqid", "room_fqid", "text_fqid"]
STUDENT_NUMERICAL = [
    "elapsed_time",
    "level",
    "page",
    "room_coor_x",
    "room_coor_y",
    "screen_coor_x",
    "screen_coor_y",
    "hover_duration",
]


def col_is_cat(col: str):
    if col in STUDENT_CATEGORICAL:
        return True
    elif col in STUDENT_NUMERICAL:
        return False
    else:
        raise ValueError(f"Unknown column {col}")


def get_aggops(col: str):
    if col_is_cat(col):
        return ["unique"]
    else:
        return ["avg", "stdSamp"]


def get_student_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"student_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    qps.append(
        StudentQP0(
            qname=f"q-{len(qps)}",
            qtype=XIPQType.NORMAL,
            data_loader=data_loader,
            verbose=verbose,
        )
    )
    cols = STUDENT_NUMERICAL + STUDENT_CATEGORICAL
    for col in cols:
        qps.append(
            StudentQP1(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.AGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=get_aggops(col),
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_performance_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False, **kwargs
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"student_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    cols = STUDENT_NUMERICAL + STUDENT_CATEGORICAL
    nf = kwargs.get("nf", len(cols))

    for i in range(nf):
        col = cols[i]
        if col in STUDENT_NUMERICAL:
            qps.append(
                StudentQP1(
                    qname=f"q-{len(qps)}",
                    qtype=XIPQType.AGG,
                    data_loader=data_loader,
                    dcol=col,
                    dcol_ops=get_aggops(col),
                    verbose=verbose,
                )
            )
        else:
            qps.append(
                StudentQP1(
                    qname=f"q-{len(qps)}",
                    qtype=XIPQType.ExactAGG,
                    data_loader=data_loader,
                    dcol=col,
                    dcol_ops=get_aggops(col),
                    verbose=verbose,
                )
            )

    for i in range(nf, len(cols)):
        col = cols[i]
        qps.append(
            StudentQP1(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.ExactAGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=get_aggops(col),
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_studentqno_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False, **kwargs
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"student_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    cols = STUDENT_NUMERICAL + STUDENT_CATEGORICAL
    nf = kwargs.get("nf", len(cols))

    for i in range(nf):
        col = cols[i]
        qps.append(
            StudentQP1(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.AGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=get_aggops(col),
                verbose=verbose,
            )
        )

    for i in range(nf, len(cols)):
        col = cols[i]
        qps.append(
            StudentQP1(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.ExactAGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=get_aggops(col),
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_studentqno_median_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False, **kwargs
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"student_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    cols = STUDENT_NUMERICAL + STUDENT_CATEGORICAL
    nf = kwargs.get("nf", len(cols))

    for i in range(nf):
        col = cols[i]
        qps.append(
            StudentQP1(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.AGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=["median" if op == "avg" else op for op in get_aggops(col)],
                verbose=verbose,
            )
        )

    for i in range(nf, len(cols)):
        col = cols[i]
        qps.append(
            StudentQP1(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.ExactAGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=["median" if op == "avg" else op for op in get_aggops(col)],
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


class StudentQP1SimMedian(StudentQP1):
    def feature_transformation(
        self, request: XIPRequest, fvec: XIPFeatureVec
    ) -> XIPFeatureVec:
        return self.feature_transformation_offset(request, fvec)


def get_studentqno_simmedian_engine(
    nparts: int, ncores: int = 0, seed: int = 0, verbose: bool = False, **kwargs
):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database=f"xip_{seed}",
        table=f"student_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    cols = STUDENT_NUMERICAL + STUDENT_CATEGORICAL
    nf = kwargs.get("nf", len(cols))

    for i in range(nf):
        col = cols[i]
        qps.append(
            StudentQP1(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.AGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=["median" if op == "avg" else op for op in get_aggops(col)],
                verbose=verbose,
            )
        )

    for i in range(nf, len(cols)):
        col = cols[i]
        qps.append(
            StudentQP1(
                qname=f"q-{len(qps)}",
                qtype=XIPQType.ExactAGG,
                data_loader=data_loader,
                dcol=col,
                dcol_ops=["median" if op == "avg" else op for op in get_aggops(col)],
                verbose=verbose,
            )
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
