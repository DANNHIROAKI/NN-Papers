import pandas as pd

from apxinfer.core.model import XIPModel, XIPRegressor
from apxinfer.core.trainer import XIPTrainer
from apxinfer.core.config import TrainerArgs, DIRHelper


class BatteryTrainer(XIPTrainer):
    def build_model(self, X: pd.DataFrame, y: pd.Series) -> XIPModel:
        if self.model_name == "mlp":
            self.logger.info(
                f"Building pipeline for {self.model_type} {self.model_name}"
            )
            from sklearn.neural_network import MLPRegressor

            model = XIPRegressor(
                MLPRegressor(
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


if __name__ == "__main__":
    args = TrainerArgs().parse_args()
    model_name = args.model
    model_type = "classifier"
    working_dir = DIRHelper.get_prepare_dir(args)

    trainer = BatteryTrainer(
        working_dir,
        model_type,
        model_name,
        args.model_seed,
        scaler_type=args.scaler_type,
        multi_class=args.multiclass,
    )
    trainer.run()
