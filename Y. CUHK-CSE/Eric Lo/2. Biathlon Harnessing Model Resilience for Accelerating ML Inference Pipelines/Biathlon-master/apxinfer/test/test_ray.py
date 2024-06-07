from typing import List, Tuple
import logging
import pandas as pd
import time
import itertools
import ray

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec, QueryCostEstimation
from apxinfer.core.utils import merge_fvecs

from apxinfer.core.query import XIPQuryProcessorRayWrapper
from apxinfer.core.query import XIPQProfile

from apxinfer.test.test_query import get_request, get_dloader
from apxinfer.test.test_query import get_qcfgs
from apxinfer.test.test_fengine import get_fengine, FEngineTestArgs


@ray.remote
class WrapperClass:
    def __init__(self, cls, *args, **kwargs):
        self.instance = cls(*args, **kwargs)

    def call_method(self, method_name, *args, **kwargs):
        method = getattr(self.instance, method_name)
        return method(*args, **kwargs)

    def get_attr(self, attr_name):
        return getattr(self.instance, attr_name)


class XIPRayFEngine:
    def __init__(
        self,
        queries: List[XIPQuryProcessorRayWrapper],
        ncores: int = 0,
        verbose: bool = False,
    ) -> None:
        self.queries = queries
        self.num_queries = len(queries)
        self.qnames = ray.get([qry.get_qname.remote() for qry in self.queries])
        print(self.qnames)
        assert len(self.qnames) == len(set(self.qnames)), "Query names must be unique"
        self.fnames = list(
            itertools.chain.from_iterable(ray.get([qry.get_fnames.remote() for qry in self.queries]))
        )
        assert len(self.fnames) == len(set(self.fnames)), "Feature names must be unique"
        print(f'fnames={self.fnames}')

        self.ncores = ncores
        self.verbose = verbose

        self.logger = logging.getLogger("XIPQEngine")
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def extract(
        self, request: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> Tuple[XIPFeatureVec, List[QueryCostEstimation]]:
        fvec, qprofiles = self.run_parallel(request, qcfgs)
        qcosts = [
            QueryCostEstimation(time=profile["total_time"], qcard=profile["card_est"],
                                ld_time=profile["loading_time"], cp_time=profile["computing_time"])
            for profile in qprofiles
        ]
        return fvec, qcosts

    def run_parallel(
        self, request: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> Tuple[XIPFeatureVec, List[XIPQProfile]]:
        for qcfg in qcfgs:
            qcfg.update({"loading_nthreads": self.ncores})
        fvecs = ray.get([qry.run.remote(request, qcfgs[i]) for i, qry in enumerate(self.queries)])
        qprofiles = ray.get([qry.get_last_qprofile.remote() for qry in self.queries])
        return merge_fvecs(fvecs), qprofiles


def test_ray(args: FEngineTestArgs):
    request = get_request()

    # sequential execution
    fengine = get_fengine(args, get_dloader(args.verbose))
    qcfgs = get_qcfgs(fengine.queries, args.qsample, 0, 0)
    fengine.set_exec_mode("sequential")
    st = time.time()
    ret_seq = fengine.run(request, qcfgs)
    seq_time = time.time() - st
    print(f"run sequential finished within {seq_time}")

    # async execution
    fengine = get_fengine(args, get_dloader(args.verbose))
    qcfgs = get_qcfgs(fengine.queries, args.qsample, 0, 0)
    fengine.set_exec_mode("async")
    st = time.time()
    ret_asy = fengine.run(request, qcfgs)
    asy_time = time.time() - st
    print(f"run async finished within {asy_time}")

    # ray execution
    fengine = get_fengine(args, get_dloader(args.verbose))
    qcfgs = get_qcfgs(fengine.queries, args.qsample, 0, 0)
    # create XIPFEngineRay from fengine
    new_qps = []
    for qp in fengine.queries:
        # delte dbclient first because it contains socket, which can not be pickled
        qp.data_loader = None # it contains socket, which can not be pickled
        qp.dbclient = None # it contains socket, which can not be pickled
        new_qps.append(XIPQuryProcessorRayWrapper.remote(qp))
    # add dbclient back, dbloader is useless, because data is already loaded
    _ = ray.get([qry.set_dbclient.remote() for qry in new_qps])
    fengine_ray = XIPRayFEngine(new_qps, args.ncores, verbose=args.verbose)
    st = time.time()
    ret_ray = fengine_ray.extract(request, qcfgs)
    ray_time = time.time() - st
    print(f"run ray finished within {ray_time}")

    fnames = ret_seq[0]["fnames"]
    fnames = ["_".join(fname.split("_")[-2:]) for fname in fnames]
    seq_fvals = ret_seq[0]["fvals"].tolist()
    asy_fvals = ret_asy[0]["fvals"].tolist()
    ray_fvals = ret_ray[0]["fvals"].tolist()
    # combine into pd.DataFrame
    df = pd.DataFrame([seq_fvals, asy_fvals, ray_fvals], columns=fnames, index=["seq", "asy", "ray"])
    # add time column to df
    df["time"] = [seq_time, asy_time, ray_time]
    print(df)


if __name__ == "__main__":
    args = FEngineTestArgs().parse_args()
    print(f"run with args: {args}")
    test_ray(args)
