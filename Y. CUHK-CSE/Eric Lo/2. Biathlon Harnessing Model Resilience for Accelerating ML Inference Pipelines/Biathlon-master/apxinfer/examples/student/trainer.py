import numpy as np
import pandas as pd
import os
import os.path as osp
from typing import List
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import xgboost as xgb

from apxinfer.core.model import XIPModel, XIPClassifier
from apxinfer.core.trainer import XIPTrainer
from apxinfer.core.config import TrainerArgs, DIRHelper


class StudentModel(BaseEstimator):
    def __init__(
        self, models: List[BaseEstimator], gidx: int, thr: float = 0.5
    ) -> None:
        self.models = models
        self.gidx = gidx
        self.thr = thr
        self.set_fimportance()

    def set_fimportance(self):
        fimps_list = []
        for model in self.models:
            try:
                fimps = model.feature_importances_
                if len(fimps.shape) > 1:
                    fimps = fimps[0]
                fimps_list.append(fimps)
            except AttributeError:
                fimps_list.append(np.zeros((model.n_features_,)))
        self.feature_importances_ = np.mean(fimps_list, axis=0)
        # add one for gidx
        self.feature_importances_ = np.insert(
            self.feature_importances_, self.gidx, 0.0
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_preds = self.predict(X)
        return f1_score(y_preds, y, average="macro")

    def predict(self, X) -> np.ndarray:
        yproba = self.predict_proba(X)[:, 1]
        return yproba > self.thr

    def predict_proba(self, X) -> np.ndarray:
        qno = X[0][self.gidx]
        if isinstance(X, list):
            X = np.array(X)
        # print(f'gidx={self.gidx}, qno={qno}, {type(X)}')
        if np.all(X[:, self.gidx] == qno):
            return self.models[int(qno) - 1].predict_proba(np.delete(X, self.gidx, axis=1))
        else:
            # X \in (n, d), where X[:, gidx] \in (1, 18)
            # each row of X should be predicted with corresponding model
            yproba = np.zeros((X.shape[0], 2), dtype=np.float32)
            for qno in range(1, 18 + 1):
                row_index = np.abs(X[:, self.gidx] - qno) < 1e-6
                if not np.any(row_index):
                    continue
                partial_preds = self.models[qno - 1].predict_proba(
                    np.delete(X[row_index], self.gidx, axis=1))
                yproba[row_index, :] = partial_preds
            return yproba


class StudentTrainer(XIPTrainer):
    def build_model(self, X: pd.DataFrame, y: pd.Series) -> XIPModel:
        if self.model_name in ["gbm", "rf", "lgbm", "xgb"]:
            self.logger.info(
                f"Building pipeline for {self.model_type} {self.model_name}"
            )
            gidx_name = "f_NORMAL_req_qno_0_q-0"
            gidx = X.columns.get_loc(gidx_name)
            fnames = X.columns.to_list()
            label_name = y.name
            valid_set = pd.read_csv(
                osp.join(self.working_dir, "dataset", "valid_set.csv")
            )
            valid_X = valid_set[fnames].values
            valid_true = valid_set[label_name].values
            X_y = pd.concat([X, y], axis=1)
            models = []
            for qno in tqdm(range(1, 18 + 1), desc="Building models"):
                if self.model_name == "gbm":
                    model = GradientBoostingClassifier(
                        n_estimators=300,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=123456,
                        verbose=True,
                    )
                elif self.model_name == "rf":
                    model = RandomForestClassifier(n_jobs=-1, random_state=0)
                elif self.model_name == "lgbm":
                    model = LGBMClassifier(random_state=0)
                elif self.model_name == "xgb":
                    xgb_params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'learning_rate': 0.05,
                        'max_depth': 4,
                        'n_estimators': 1000,
                        'early_stopping_rounds': 50,
                        'tree_method': 'hist',
                        'subsample': 0.8,
                        'colsample_bytree': 0.4,
                        'use_label_encoder': False,
                        'random_state': 0
                    }
                    model = xgb.XGBClassifier(**xgb_params)
                else:
                    raise NotImplementedError
                X_qno = X_y[X_y[gidx_name] == qno]
                X_fit = X_qno[fnames].drop(columns=[gidx_name]).values
                y_fit = X_qno[label_name].values
                if self.model_name == "xgb":
                    valid_X_qno = valid_set[valid_set[gidx_name] == qno]
                    X_test = valid_X_qno[fnames].drop(columns=[gidx_name]).values
                    y_test = valid_X_qno[label_name].values
                    model.fit(X_fit, y_fit, eval_set=[(X_test, y_test)], verbose=0)
                else:
                    model.fit(X_fit, y_fit)
                models.append(model)
                yproba = model.predict_proba(X_fit)
                print(f"QNO {qno} F1: {f1_score(y_fit, yproba[:, 1] > 0.5):.04f}")

            model = XIPClassifier(
                StudentModel(models=models, gidx=X.columns.get_loc(gidx_name))
            )
            valid_proba = model.predict_proba(valid_X)[:, 1]

            scores, thresholds = [], []
            best_score = model.score(valid_set[fnames].values, valid_true)
            best_threshold = 0.5
            for threshold in tqdm(
                np.arange(0.4, 0.81, 0.01), desc="Finding best threshold"
            ):
                preds = (valid_proba.reshape((-1)) > threshold).astype("int")
                m = f1_score(preds, valid_true, average="macro")
                scores.append(m)
                thresholds.append(threshold)
                if m > best_score:
                    best_score = m
                    best_threshold = threshold
            # print thresholds, scores as dataframe
            df = pd.DataFrame({"threshold": thresholds, "score": scores})
            print(df)
            print(f"Best threshold: {best_threshold:.02f} with score {best_score:.04f}")

            model = XIPClassifier(
                StudentModel(models=models, gidx=gidx, thr=best_threshold)
            )
            return model
        elif self.model_name == "tfgbm":
            import tensorflow as tf
            import tensorflow_addons as tfa
            import tensorflow_decision_forests as tfdf

            gbtm = tfdf.keras.GradientBoostedTreesModel(verbose=0)
            gbtm.compile(metrics=["accuracy"])
            model = gbtm
        else:
            return super().build_model(X, y)


class StudentQNoTrainer(XIPTrainer):
    def build_model(self, X: pd.DataFrame, y: pd.Series) -> XIPModel:
        if self.model_name == "mlp":
            self.logger.info(
                f"Building pipeline for {self.model_type} {self.model_name}"
            )
            from sklearn.neural_network import MLPClassifier

            model = XIPClassifier(
                MLPClassifier(
                    hidden_layer_sizes=(100, 100),
                    random_state=self.seed,
                    learning_rate_init=0.01,
                    max_iter=1000,
                    verbose=True,
                ),
                multi_class=self.multi_class,
            )
            model.fit(X.values, y.values)
            return model
        else:
            return super().build_model(X, y)


if __name__ == "__main__":
    args = TrainerArgs().parse_args()
    model_name = args.model
    model_type = "classifier"
    seed = args.seed
    working_dir = DIRHelper.get_prepare_dir(args)

    trainer = StudentQNoTrainer(
        working_dir,
        model_type,
        model_name,
        seed,
        scaler_type=args.scaler_type,
        multi_class=args.multiclass,
    )
    trainer.run()
