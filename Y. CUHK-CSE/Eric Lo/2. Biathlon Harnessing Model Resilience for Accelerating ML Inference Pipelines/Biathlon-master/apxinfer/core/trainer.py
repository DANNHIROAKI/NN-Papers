import numpy as np
import pandas as pd
import os
import os.path as osp
import joblib
import logging
import json
from sklearn import metrics

from apxinfer.core.config import DIRHelper
from apxinfer.core.model import XIPModel, create_model, evaluate_model

logging.basicConfig(level=logging.INFO)


class XIPTrainer:
    """This Worker prepares dataset for model training and evaluation,
    as well as training and evaluating the model
    """

    def __init__(
        self,
        working_dir: str,
        model_type: str,
        model_name: str,
        seed: int,
        scaler_type: str = None,
        multi_class: bool = False,
    ) -> None:
        self.working_dir = working_dir

        self.model_type = model_type
        self.model_name = model_name
        self.scaler_type = scaler_type
        self.multi_class = multi_class

        self.seed = seed
        self.logger = logging.getLogger("XIPTrainer")

        self.model_dir = osp.join(self.working_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_tag = DIRHelper.get_model_tag(self.model_name, self.scaler_type)

    def build_model(self, X: pd.DataFrame, y: pd.Series) -> XIPModel:
        self.logger.info(f"Building pipeline for {self.model_type} {self.model_name}")
        model = create_model(
            self.model_type,
            self.model_name,
            scaler_type=self.scaler_type,
            multi_class=self.multi_class,
            random_state=self.seed,
        )
        if isinstance(X, pd.DataFrame):
            model.fit(X.values, y.values)
        else:
            model.fit(X, y)
        return model

    def run(self) -> None:
        train_set = pd.read_csv(osp.join(self.working_dir, "dataset", "train_set.csv"))
        valid_set = pd.read_csv(osp.join(self.working_dir, "dataset", "valid_set.csv"))
        test_set = pd.read_csv(osp.join(self.working_dir, "dataset", "test_set.csv"))

        cols = list(train_set.columns)
        fnames = [col for col in cols if col.startswith("f_")]
        # label_name = cols[-1]
        label_name = "label"

        model = self.build_model(train_set[fnames], train_set[label_name])
        joblib.dump(model, osp.join(self.model_dir, f"{self.model_tag}.pkl"))

        train_pred = model.predict(train_set[fnames].values)
        valid_pred = model.predict(valid_set[fnames].values)
        test_pred = model.predict(test_set[fnames].values)

        # save evaluations
        self.logger.info(f"Saving evaluations for {self.model_type} {self.model_tag}")
        train_evals = evaluate_model(
            model, train_set[fnames].values, train_set[label_name].values
        )
        valid_evals = evaluate_model(
            model, valid_set[fnames].values, valid_set[label_name].values
        )
        test_evals = evaluate_model(
            model, test_set[fnames].values, test_set[label_name].values
        )
        all_evals = {"train": train_evals, "valid": valid_evals, "test": test_evals}
        with open(osp.join(self.model_dir, f"{self.model_tag}_evals.json"), "w") as f:
            json.dump(all_evals, f, indent=4)
        self.logger.info(all_evals)

        # for classification pipeline, we print and save the classification report
        if self.model_type == "classification":
            self.logger.info(
                f"Saving classification reports for {self.model_type} {self.model_tag}"
            )
            train_report = metrics.classification_report(
                train_set[label_name], train_pred
            )
            valid_report = metrics.classification_report(
                valid_set[label_name], valid_pred
            )
            test_report = metrics.classification_report(test_set[label_name], test_pred)
            self.logger.info(train_report)
            self.logger.info(valid_report)
            self.logger.info(test_report)
            with open(
                osp.join(
                    self.working_dir,
                    "model",
                    f"{self.model_tag}_classification_reports.txt",
                ),
                "w",
            ) as f:
                f.write(f"train_report: \n{train_report}\n")
                f.write(f"valid_report: \n{valid_report}\n")
                f.write(f"test_report: \n{test_report}\n")

        # save global feature importance of the model
        self.logger.info(
            f"Calculating feature importance for {self.model_type} {self.model_tag}"
        )
        feature_importance = model.get_feature_importances(
            X_val=valid_set[fnames].values, y_val=valid_set[label_name].values
        )
        gfimps: pd.DataFrame = pd.DataFrame(
            {"fname": fnames, "importance": feature_importance},
            columns=["fname", "importance"],
        )
        gfimps.to_csv(
            osp.join(self.model_dir, f"{self.model_tag}_feature_importance.csv"),
            index=False,
        )
        self.logger.info(gfimps)

        self.logger.info("Calculating permutation feature importance")
        perm_fimps = model.get_permutation_fimps(
            X_val=valid_set[fnames].values, y_val=valid_set[label_name].values
        )
        pfimps: pd.DataFrame = pd.DataFrame(
            {"fname": fnames, "importance": perm_fimps},
            columns=["fname", "importance"],
        )
        pfimps.to_csv(
            osp.join(self.model_dir, f"{self.model_tag}_pfimps.csv"),
            index=False,
        )
        self.logger.info(pfimps)
