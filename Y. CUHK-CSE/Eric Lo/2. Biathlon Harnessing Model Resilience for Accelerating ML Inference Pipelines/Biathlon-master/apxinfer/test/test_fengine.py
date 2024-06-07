from tap import Tap
import time
import pandas as pd

from apxinfer.core.data import XIPDataLoader
from apxinfer.core.fengine import XIPFEngine

from apxinfer.test.test_query import get_request, get_dloader
from apxinfer.test.test_query import get_qps, get_qcfgs


class FEngineTestArgs(Tap):
    verbose: bool = False
    qsample: float = 0.1
    ld_nthreads: int = 1
    cp_nthreads: int = 0
    ncores: int = 1


def get_fengine(args: FEngineTestArgs, data_loader: XIPDataLoader):
    qps = get_qps(data_loader, verbose=args.verbose)
    fengine = XIPFEngine(qps, args.ncores, verbose=args.verbose)
    return fengine


if __name__ == "__main__":
    args = FEngineTestArgs().parse_args()
    print(f"run with args: {args}")

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
    fengine.set_exec_mode("parallel")
    st = time.time()
    ret_ray = fengine.run(request, qcfgs)
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
