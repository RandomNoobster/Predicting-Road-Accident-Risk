from sklearn.linear_model import LinearRegression, LassoCV
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras
from tf_keras import layers
import numpy as np
import xgboost as xgb
from util.model_utils import PersistentMixin

class BaselineModel(PersistentMixin):
    """
    We start with a Simple Linear Regressor as our initial hypothesis.
    
    In accordance with the principle of "Occam's Razor" discussed in the 
    'A Baseline Approach' slides, our goal is to establish a solution with the 
    simplest possible model. This helps us determine if the data actually 
    requires non-linear complexity.
    
    For uncertainty, we assume 'Homoscedasticity'—the idea that the model's 
    error is constant across all inputs. We estimate this global noise 
    level by calculating the standard deviation of the training residuals.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.std_err = 0.0

    def fit(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        # We calculate the standard deviation of the error (residuals).
        # This represents our uncertainty as a single constant value.
        self.std_err = np.std(y - preds)

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        mean = self.predict(X)
        # We broadcast the constant error estimate across all predictions.
        stddev = np.full_like(mean, self.std_err)
        return mean, stddev


class L1Model(PersistentMixin):
    """
    We use Lasso (L1 Regularization) to refine our baseline through feature selection.
    
    As noted in the 'Lasso' section of the baseline slides, simple OLS regression 
    often assigns non-zero weights to irrelevant features, which can lead to overfitting. 
    By using an L1 penalty, we force the weights of less important features to zero. 
    This creates a 'sparse' model that is easier for us to interpret and present 
    to a domain expert.
    """

    def __init__(self, cv=15, random_state=42):
        # We use LassoCV with 15-fold cross-validation to ensure our 
        # regularization strength (alpha) is robustly calibrated.
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
    We implement XGBoost to capture non-linearities and feature interactions.
    
    Compared to our linear models, the 'Non-Linear Models' slides highlight 
    that Gradient Boosted Trees can model complex dependencies (like feature A 
    only being relevant when feature B is high) which linear regressors miss.
    
    A major limitation of XGBoost is that it is a point estimator—it typically 
    only predicts a single value. To provide uncertainty, we implement 
    Quantile Regression. We train three independent versions of our model:
      1. One for the mean (MSE loss).
      2. One for the 16th percentile (lower bound).
      3. One for the 84th percentile (upper bound).
      
    This 16-84 range corresponds to approximately +/- 1 standard deviation 
    if we assume the underlying noise is normally distributed.
    """

    def __init__(
        self,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42,
        objective=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        if objective is not None:
            self.mean_objective = objective.mean
            self.lower_quantile_objective = objective.lower_quantile
            self.upper_quantile_objective = objective.upper_quantile
        else:
            self.mean_objective = "reg:squarederror"
            self.lower_quantile_objective = "reg:quantileerror"
            self.upper_quantile_objective = "reg:quantileerror"

        # Primary model for point predictions
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            objective=self.mean_objective,
        )

        # Quantile models used to build a confidence interval
        self.model_lower = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            objective=self.lower_quantile_objective,
            quantile_alpha=0.16,
        )

        self.model_upper = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            objective=self.upper_quantile_objective,
            quantile_alpha=0.84,
        )

    def fit(self, X, y, eval_set=None, verbose=False):
        # We accept a 3x increase in training time to gain the ability
        # to quantify how confident our non-linear model is.
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)
        self.model_lower.fit(X, y, eval_set=eval_set, verbose=verbose)
        self.model_upper.fit(X, y, eval_set=eval_set, verbose=verbose)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        mean = self.model.predict(X)
        lower = self.model_lower.predict(X)
        upper = self.model_upper.predict(X)

        # Since the distance from 16% to 84% covers 2 standard deviations, 
        # we divide the spread by 2 to get our estimated sigma.
        stddev = (upper - lower) / 2.0

        # Regarding the clamping: trees are trained independently. In sparse
        # data regions, it's possible for the 'lower' tree to predict a value 
        # higher than the 'upper' tree due to sampling noise. Since a 
        # negative standard deviation is mathematically impossible, we use
        # np.maximum to ensure numerical stability. This isn't "lying," but 
        # rather correcting for the lack of coordination between the trees.
        stddev = np.maximum(stddev, 1e-6)

        return mean, stddev


class NeuroProbabilisticModel(PersistentMixin):
    """
    We implement a Neuro-Probabilistic model to estimate Aleatoric Uncertainty.
    
    Following the 'Neuro-Probabilistic Models' lecture, we distinguish 
    between uncertainty in the model (Epistemic) and uncertainty inherent 
    to the data/process (Aleatoric). This model focuses on Aleatoric 
    uncertainty by learning a conditional distribution: y ~ N(mu(x), sigma(x)).
    
    Unlike our baselines, this model is 'Heteroscedastic'—it can learn that 
    some input regions are noisier than others, which is critical for 
    industrial risk assessment.
    """

    def __init__(self, input_shape=(16,), hidden_layers=[32, 32], custom_loss=None):
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers

        model_in = keras.Input(shape=input_shape, dtype="float32")
        x = model_in
        for h in hidden_layers:
            x = layers.Dense(h, activation="relu")(x)

        # Our output layer has 2 neurons: one for the mean and one for the scale.
        params = layers.Dense(2, activation="linear")(x)

        # In accordance with the slides on 'Building a Neuro-Probabilistic Model',
        # we use a DistributionLambda to wrap our distribution logic.
        def distribution_builder(t):
            # We apply softplus to the scale neuron because the standard 
            # deviation MUST be positive. We add a small epsilon (1e-3) 
            # to avoid numerical issues during log-likelihood calculations.
            return tfp.distributions.Normal(
                loc=t[..., :1], scale=1e-3 + tf.math.softplus(t[..., 1:])
            )

        model_out = tfp.layers.DistributionLambda(distribution_builder)(params)
        self.model = keras.Model(model_in, model_out)

        # We minimize the Negative Log Likelihood (NLL). As the course explains,
        # NLL is the proper way to train models that output distributions,
        # as it rewards the model for placing high probability on the true values.
        negloglikelihood = lambda y_true, dist: -dist.log_prob(y_true)
        
        # If a custom loss is provided, we use it instead of NLL.
        loss_func = custom_loss.loss if custom_loss else negloglikelihood
        self.model.compile(optimizer="adam", loss=loss_func)

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

        # We chose a large batch size (2048) for two reasons:
        # 1. To improve computational throughput (faster training).
        # 2. To stabilize the gradients. Probabilistic models can have 
        # noisy gradients when estimating variance; larger batches provide 
        # a better estimate of the distribution's properties in each step.
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
        # Calling the model returns a distribution object.
        dist = self.model(X_tf)
        return dist.mean().numpy().ravel()

    def predict_with_uncertainty(self, X):
        X_tf = np.asarray(X).astype("float32")
        # We call the model to obtain the distribution parameters (mu and sigma).
        dist = self.model(X_tf)
        mean = dist.mean().numpy().ravel()
        stddev = dist.stddev().numpy().ravel()
        return mean, stddev