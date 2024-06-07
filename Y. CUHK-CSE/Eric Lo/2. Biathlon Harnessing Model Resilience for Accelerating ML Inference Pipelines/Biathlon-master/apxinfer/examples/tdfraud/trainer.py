import pandas as pd

from apxinfer.core.model import XIPModel, XIPClassifier
from apxinfer.core.trainer import XIPTrainer
# from apxinfer.core.config import TrainerArgs, DIRHelper


class TDFraudTrainer(XIPTrainer):
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
                )
            )
            model.fit(X.values, y.values)
            return model
        else:
            return super().build_model(X, y)


class TDFraudKaggleTrainer(XIPTrainer):
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
                )
            )
            model.fit(X.values, y.values)
            return model
        else:
            return super().build_model(X, y)
