from tap import Tap
import importlib
import os
import numpy as np
import joblib
import time
from concurrent.futures import ProcessPoolExecutor

from apxinfer.core.prepare import XIPPrepareWorker
from apxinfer.core.config import DIRHelper, LoadingHelper
from apxinfer.core.config import BaseXIPArgs
from apxinfer.core.config import PrepareArgs, TrainerArgs
from apxinfer.core.config import OfflineArgs, OnlineArgs

from apxinfer.core.model import XIPModel
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def parallel_predict(model, X, n_jobs=-1):
    """Run predictions using multiple cores."""
    if n_jobs == -1:
        n_jobs = 4  # Automatically use the maximum number of cores if n_jobs is set to -1.
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        n_samples = len(X)
        predictions = list(executor.map(model.predict, (X[i::n_jobs] for i in range(n_jobs))))
        result = []
        for i in range(n_samples):
            result.append(predictions[i % n_jobs][i // n_jobs])
    return result


if __name__ == "__main__":
    # python test_model.py --task final/tripsfeast --model lgbm
    # python test_model.py --task final/tick --model lr
    # python test_model.py --task final/machinery --model mlp
    # python test_model.py --task final/machinery --model knn
    # python test_model.py --task final/machinerymulti --model svm
    # python test_model.py --task final/tdfraud --model xgb
    batch_size = 1000
    nrounds = 10
    args = OnlineArgs().parse_args(known_only=True)
    test_set = LoadingHelper.load_dataset(
        args, "test", args.nreqs, offset=args.nreqs_offset
    )
    verbose = args.verbose and len(test_set) <= 10
    model: XIPModel = LoadingHelper.load_model(args)

    dataset = test_set
    cols = dataset.columns.tolist()
    req_cols = [col for col in cols if col.startswith("req_")]
    fcols = [col for col in cols if col.startswith("f_")]
    label_col = "label"

    requests = dataset[req_cols].to_dict(orient="records")
    labels = dataset[label_col].to_numpy()
    ext_features = dataset[fcols].to_numpy()

    # if args.model == "knn":
    #     model = KNeighborsClassifier(n_jobs=-1)
    #     model.fit(ext_features, labels)

    X = np.repeat(ext_features, repeats=1 + (batch_size // len(ext_features)), axis=0)
    X = X[:batch_size]
    st = time.time()
    for i in range(nrounds):
        model.predict(X).astype(np.float64)
    et = time.time()
    total_time = et - st
    print(f'total_time: {total_time}')
    avg_time = total_time / nrounds
    avg_inf_time = avg_time / len(X)
    print(f"{nrounds} rounds avg time cost: {avg_time:.4f} / {len(X)} = {avg_inf_time}")

    # st = time.time()
    # for i in range(nrounds):
    #     with joblib.parallel_backend(backend='loky', n_jobs=4):
    #         y = model.predict(X).astype(np.float64)
    # et = time.time()
    # total_time = et - st
    # print(f'total_time: {total_time}')
    # avg_time = total_time / nrounds
    # avg_inf_time = avg_time / len(X)
    # print(f"{nrounds} rounds avg time cost: {avg_time:.4f} / {len(X)} = {avg_inf_time}")
