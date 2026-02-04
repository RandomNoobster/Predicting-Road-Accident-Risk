from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.base import BaseEstimator
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras
from tf_keras import layers
import numpy as np
import joblib
import json
import os
import xgboost as xgb
from pathlib import Path

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class PersistentMixin:
    def fit_or_load(self, X, y, **fit_params):
        init_params = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k != "model"
        }

        config = {
            "class": self.__class__.__name__,
            "init": init_params,
            "fit": fit_params,
        }
        unique_id = joblib.hash(config)

        base_name = f"{self.__class__.__name__}_{unique_id}"
        is_keras = hasattr(self.model, "save_weights")

        filename = f"{base_name}.weights.h5" if is_keras else f"{base_name}.pkl"
        file_path = MODEL_DIR / filename
        meta_path = MODEL_DIR / f"{base_name}.json"

        if file_path.exists():
            print(f"✅ Found cached model: {filename}")
            if is_keras:
                self.model.load_weights(file_path)
            else:
                loaded_obj = joblib.load(file_path)
                self.__dict__.update(loaded_obj.__dict__)
            return self

        print(f"⚙️  Training new model: {base_name}...")
        self.fit(X, y, **fit_params)

        if is_keras:
            self.model.save_weights(file_path)
        else:
            joblib.dump(self, file_path)

        with open(meta_path, "w") as f:
            json.dump(config, f, indent=4, default=str)

        return self


class BaselineModel(PersistentMixin):
    """
    Baseline regression model using Linear Regression.
    Uncertainty is estimated as the standard deviation of training residuals.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.std_err = 0.0

    def fit(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.std_err = np.std(y - preds)

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        mean = self.predict(X)
        stddev = np.full_like(mean, self.std_err)
        return mean, stddev


class L1Model(PersistentMixin):
    """
    L1-regularized regression model using LassoCV.
    Uncertainty is estimated as the standard deviation of training residuals.
    """

    def __init__(self, cv=15, random_state=42):
        self.cv = cv
        self.random_state = random_state
        self.model = LassoCV(cv=cv, random_state=random_state)
        self.std_err = 0.0

    def fit(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.std_err = np.std(y - preds)

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        mean = self.predict(X)
        stddev = np.full_like(mean, self.std_err)
        return mean, stddev


class XGBoostModel(PersistentMixin):
    """
    Non-linear regression using Gradient Boosted Trees (XGBoost).

    Curriculum Ref: 'Non-Linear Models' (Gradient Boosting).

    Uncertainty Quantification:
    Since standard XGBoost is a point estimator, we implement 'Quantile Regression'
    to estimate uncertainty. We train 3 separate estimators:
      1. Mean (Objective: Squared Error)
      2. Lower Bound (Objective: Quantile 0.16)
      3. Upper Bound (Objective: Quantile 0.84)

    This range (16%-84%) approximates +/- 1 Standard Deviation.
    """

    def __init__(
        self,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Main model (Mean/MSE)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="reg:squarederror",
        )

        # Lower Bound Model (16th Percentile ~ -1 Sigma)
        self.model_lower = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="reg:quantileerror",
            quantile_alpha=0.16,
        )

        # Upper Bound Model (84th Percentile ~ +1 Sigma)
        self.model_upper = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="reg:quantileerror",
            quantile_alpha=0.84,
        )

    def fit(self, X, y, eval_set=None, verbose=False):
        # We need to fit all three models
        # Note: In a production setting, this triples training time.

        # 1. Fit Main
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

        # 2. Fit Quantiles (Uncertainty)
        self.model_lower.fit(X, y, eval_set=eval_set, verbose=verbose)
        self.model_upper.fit(X, y, eval_set=eval_set, verbose=verbose)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        """
        Returns:
            mean: The prediction from the main MSE model.
            stddev: Approximated as (Upper_Quantile - Lower_Quantile) / 2
        """
        mean = self.model.predict(X)
        lower = self.model_lower.predict(X)
        upper = self.model_upper.predict(X)

        # Approximate Standard Deviation from the quantile spread
        # (Q84 - Q16) covers roughly 2 standard deviations in a normal dist.
        stddev = (upper - lower) / 2.0

        # Safety: Ensure stddev is non-negative (trees might cross in sparse regions)
        stddev = np.maximum(stddev, 1e-6)

        return mean, stddev


class NeuroProbabilisticModel(PersistentMixin):
    """
    NPN for continuous data using a Normal distribution.
    This estimates both the risk value and the input-specific uncertainty.
    """

    def __init__(self, input_shape=(16,), hidden_layers=[32, 32]):
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers

        model_in = keras.Input(shape=input_shape, dtype="float32")
        x = model_in
        for h in hidden_layers:
            x = layers.Dense(h, activation="relu")(x)

        params = layers.Dense(2, activation="linear")(x)

        def distribution_builder(t):
            return tfp.distributions.Normal(
                loc=t[..., :1], scale=1e-3 + tf.math.softplus(t[..., 1:])
            )

        model_out = tfp.layers.DistributionLambda(distribution_builder)(params)
        self.model = keras.Model(model_in, model_out)

        negloglikelihood = lambda y_true, dist: -dist.log_prob(y_true)
        self.model.compile(optimizer="adam", loss=negloglikelihood)

    def fit(
        self, X, y, validation_data=None, epochs=50, batch_size=2048, verbose="auto"
    ):
        X_tf = np.asarray(X).astype("float32")
        y_tf = np.asarray(y).astype("float32")

        val_data = None
        if validation_data is not None:
            X_v, y_v = validation_data
            val_data = (
                np.asarray(X_v).astype("float32"),
                np.asarray(y_v).astype("float32"),
            )

        return self.model.fit(
            X_tf,
            y_tf,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def predict(self, X):
        X_tf = np.asarray(X).astype("float32")
        dist = self.model(X_tf)
        return dist.mean().numpy().ravel()

    def predict_with_uncertainty(self, X):
        X_tf = np.asarray(X).astype("float32")
        dist = self.model(X_tf)
        mean = dist.mean().numpy().ravel()
        stddev = dist.stddev().numpy().ravel()
        return mean, stddev
