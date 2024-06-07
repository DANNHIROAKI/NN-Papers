import pandas as pd
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend
from lightgbm import LGBMClassifier

from apxinfer.core.model import XIPModel, XIPClassifier
from apxinfer.core.trainer import XIPTrainer


class CheapTripsTrainer(XIPTrainer):
    def build_model(self, X: pd.DataFrame, y: pd.Series) -> XIPModel:
        # building lgbm with grid search cv
        if self.model_name == "lgbm-tune":
            grid = {
                "learning_rate": [0.01, 0.1, 0.5],
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, 15],
                "num_leaves": [10, 20, 30],
            }
            return self.build_tunned_model(X, y, params=grid, cv=5, n_jobs=1, verbose=0)
        elif self.model_name == "mlp":
            self.logger.info(
                f"Building pipeline for {self.model_type} {self.model_name}"
            )
            from sklearn.neural_network import MLPClassifier

            model = XIPClassifier(
                MLPClassifier(
                    hidden_layer_sizes=(100, 50, 100),
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

    def build_tunned_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> XIPModel:
        self.logger.info(f"Building pipeline for {self.model_type} {self.model_name}")
        if kwargs.get('tunner', 'grid') == 'grid':
            tunner = GridSearchCV(
                LGBMClassifier(
                    random_state=self.seed,
                    n_jobs=kwargs.get('n_jobs', 1),
                    verbose=kwargs.get('verbose', 0),
                ),
                kwargs.get('params', {}),
                cv=kwargs.get('cv', 5),
                n_jobs=kwargs.get('n_jobs', 1),
                verbose=kwargs.get('verbose', 0),
            )
        else:
            raise ValueError(f"Unsupported tunner: {kwargs.get('tunner', 'grid')}")
        with parallel_backend('threading'):
            tunner.fit(X.values, y.values)
        model = XIPClassifier(tunner.best_estimator_)
        return model
