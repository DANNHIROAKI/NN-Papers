import pandas as pd
import numpy as np

from apxinfer.core.model import XIPModel, XIPRegressor
from apxinfer.core.trainer import XIPTrainer
from apxinfer.core.config import TrainerArgs, DIRHelper

from keras.utils import set_random_seed
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


class TickTrainer(XIPTrainer):
    def build_model(self, X: pd.DataFrame, y: pd.Series) -> XIPModel:
        if self.model_name == "mlp":
            self.logger.info(
                f"Building pipeline for {self.model_type} {self.model_name}"
            )
            from sklearn.neural_network import MLPRegressor

            model = XIPRegressor(
                MLPRegressor(
                    hidden_layer_sizes=(100, 50, 100),
                    random_state=self.seed,
                    learning_rate_init=0.01,
                    max_iter=1000,
                    verbose=True,
                )
            )
            model.fit(X.values, y.values)
            return model
        elif self.model_name == "lstm":
            self.logger.info(
                f"Building pipeline for {self.model_type} {self.model_name}"
            )

            # create and fit the LSTM network
            set_random_seed(self.seed)
            batch_size = 4
            nbatches = X.shape[0] // batch_size
            X = X[:nbatches*batch_size]
            y = y[:nbatches*batch_size]
            model = Sequential()
            model.add(LSTM(4, input_shape=(batch_size, X.shape[1])))
            model.add(Dense(1))
            model.compile(loss="mean_squared_error", optimizer="adam")
            X = np.reshape(
                X.values,
                (X.values.shape[0] // batch_size, batch_size, X.values.shape[1]),
            )
            y = np.reshape(y.values, (y.values.shape[0] // batch_size, batch_size, 1))
            # callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            model.fit(X, y, epochs=500, batch_size=batch_size, verbose=2)
            model = XIPRegressor(model)
            model.predict = model.keras_predict
            model.score = model.keras_score
            return model
        else:
            return super().build_model(X, y)


if __name__ == "__main__":
    args = TrainerArgs().parse_args()
    model_name = args.model
    model_type = "regressor"
    seed = args.seed
    working_dir = DIRHelper.get_prepare_dir(args)

    trainer = TickTrainer(
        working_dir, model_type, model_name, seed, scaler_type=args.scaler_type
    )
    trainer.run()
