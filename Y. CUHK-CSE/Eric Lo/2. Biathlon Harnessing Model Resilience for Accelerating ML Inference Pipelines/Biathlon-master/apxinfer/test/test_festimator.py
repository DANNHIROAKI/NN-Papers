import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats


from multiprocessing import Pool
from multiprocessing import shared_memory

from functools import partial


from apxinfer.core.festimator import XIPDataAggregator, XIPFeatureErrorEstimator
from apxinfer.core.config import EXP_HOME

if __name__ == "__main__":
    FIG_HOME = os.path.join(EXP_HOME, "festimator")
    os.makedirs(FIG_HOME, exist_ok=True)
    p = 0.0001
    tsize = 1000000
    agg = "min"
    bs_nresamples = 100
    nthreads = 4
    ssize = int(tsize * p)
    print(f"ssize={ssize}")
    samples = np.random.normal(10, 2, size=(ssize, 1))

    fest_signle = XIPFeatureErrorEstimator(
        bs_max_nthreads=1, bs_nresamples=bs_nresamples
    )
    fest_parallel = XIPFeatureErrorEstimator(
        bs_max_nthreads=nthreads, bs_nresamples=bs_nresamples
    )

    st = time.time()
    fest_signle.bootstrap(samples, p, tsize, agg)
    single_tcost = time.time() - st

    print(f"bs_tcost: {fest_signle.bs_tcost}")
    print(f"bs_random_tcost: {fest_signle.bs_random_tcost}")
    print(f"bs_resampling: {fest_signle.bs_resampling_tcost}")

    st = time.time()
    fest_parallel.bootstrap(samples, p, tsize, agg)
    parallel_tcost = time.time() - st

    def single_worker(samples, p, tsize, agg):
        fest = XIPFeatureErrorEstimator(bs_max_nthreads=1, bs_nresamples=bs_nresamples)
        return fest.bootstrap(samples, p, tsize, agg)

    pool = Pool(nthreads)
    st = time.time()
    worker = partial(single_worker, samples, p, tsize)
    results = pool.map(worker, [agg] * nthreads)
    pool.close()  # close the pool
    pool.join()
    singlep_tcost = time.time() - st
    print(f"psg results={[np.std(re) for re in results]}")

    pool = Pool(nthreads)
    st = time.time()
    shm = shared_memory.SharedMemory(create=True, size=samples.nbytes)
    sh_samples = np.ndarray(samples.shape, dtype=samples.dtype, buffer=shm.buf)
    sh_samples[:] = samples[:]
    print(f"shm init time: {time.time() - st}")
    worker = partial(single_worker, sh_samples, p, tsize)
    results = pool.map(worker, [agg] * nthreads)
    pool.close()  # close the pool
    pool.join()
    shm.close()
    shm.unlink()
    singlepshm_tcost = time.time() - st
    print(f"shm results={[np.std(re) for re in results]}")

    print(f"fest_signle   = {single_tcost}")
    print(f"fest_parallel = {parallel_tcost}")
    print(f"signle_parall = {singlep_tcost / nthreads}")
    print(f"signle_p_shm  = {singlepshm_tcost / nthreads}")

    print(
        "parallel queries is the best, \
          but the parallelism is limited by number of queries."
    )

    agg = "sum"
    features = XIPDataAggregator.estimate(samples, p, agg)
    fs, ferr = fest_signle.estimate(samples, p, tsize, features, agg)

    bs_ests = fest_signle.bootstrap(samples, p, tsize, agg)
    bt_fs = np.mean(bs_ests, axis=0)
    bt_bias = features - bt_fs
    bt_fs_wob = features + bt_bias
    bt_ferr = np.std(bs_ests, axis=0, ddof=1)

    print(f"real_fs   = {2}")
    print(f"fs        = {fs}")
    print(f"bt_fs     = {bt_fs}")
    print(f"bt_fs_wob = {bt_fs_wob}")

    print(f"ferr      = {ferr}")
    print(f"bt_ferr   = {bt_ferr}")

    fs_list = []
    bt_fs_list = []
    bt_fs_wob_list = []
    for i in tqdm(range(1000)):
        smp = np.random.normal(10, 2, size=(ssize, 2))
        smp += np.random.beta(10, 2, size=(ssize, 2))
        smp = smp[np.where(smp[:, 1] < 11), 0].reshape(-1, 1)
        # print(f'shape={smp.shape}')

        features_ = XIPDataAggregator.estimate(smp, p, agg)
        fs_, _ = fest_signle.estimate(smp, p, tsize, features_, agg)

        bs_ests_ = fest_signle.bootstrap(smp, p, tsize, agg)
        bt_fs_ = np.mean(bs_ests_, axis=0)
        bt_bias_ = features_ - bt_fs_
        bt_fs_wob_ = features_ + bt_bias_

        fs_list.append(fs_)
        bt_fs_list.append(bt_fs_)
        bt_fs_wob_list.append(bt_fs_wob_)

    fs_list = np.array(fs_list)
    bt_fs_list = np.array(bt_fs_list)
    bt_fs_wob_list = np.array(bt_fs_wob_list)

    fig, (axes, axes2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    axes.hist(
        bt_fs_list[:, 0], density=True, bins=20, label="bt_fs", color="g", alpha=1.0
    )
    # get twin axes sharing xaxis
    axes.hist(
        bt_fs_wob_list[:, 0],
        density=True,
        bins=20,
        label="bt_fs_wob",
        color="tab:blue",
        alpha=0.4,
    )
    bin_vs, bins, _ = axes2.hist(
        fs_list[:, 0], density=True, bins=20, label="fs", color="r", alpha=0.4
    )
    if agg == "stdPop":
        x = sorted(np.random.uniform(1.5, 2.5, size=1000))
    elif agg == "varPop":
        x = sorted(np.random.uniform(3, 5, size=1000))
    else:
        # determin x range accoridng to bins
        x = []
        for i in range(len(bins) - 1):
            x.append((bins[i] + bins[i + 1]) / 2)
        x = np.array(sorted(x))
    axes2.plot(
        x,
        stats.norm.pdf(x, loc=np.mean(fs_list), scale=ferr),
        label="d-fs",
        color="r",
        linestyle="--",
    )
    axes.plot(
        x,
        stats.norm.pdf(x, loc=np.mean(bt_fs_list), scale=bt_ferr),
        label="d-bt_fs",
        color="g",
        linestyle="--",
    )
    axes.plot(
        x,
        stats.norm.pdf(x, loc=np.mean(bt_fs_wob_list), scale=bt_ferr),
        label="d-bt_fs_wob",
        color="tab:blue",
        linestyle="--",
    )

    axes.legend()
    axes2.legend()
    plt.savefig(f"{FIG_HOME}/festimator_{agg}_{ssize}_{bs_nresamples}.pdf")
