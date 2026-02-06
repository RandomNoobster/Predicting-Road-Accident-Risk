import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

class CustomLoss():
    def __init__(self, penalty_underestimate):
        self.penalty_underestimate = penalty_underestimate

    def loss(self, y_true, y_pred):
        diff = y_pred - y_true
        weights = np.where(diff < 0, self.penalty_underestimate, 1.0)
        
        weighted_mae = np.mean(np.abs(diff) * weights)
        return weighted_mae
    

# For Neuro Probabilistic Models the loss has to be calculated
# a bit differently as they output a distribution instead of one value
class CustomNeuroLoss:
    def __init__(self, penalty_underestimate):
        self.penalty_underestimate = float(penalty_underestimate)

    def loss(self, y_true, dist):
        mean = dist.mean()

        y_true = tf.cast(y_true, mean.dtype)

        diff = mean - y_true

        penalty = tf.cast(self.penalty_underestimate, mean.dtype)
        one = tf.cast(1.0, mean.dtype)

        weights = tf.where(diff < 0.0, penalty, one)

        nll = -dist.log_prob(y_true)

        weighted_nll = nll * weights

        return tf.reduce_mean(weighted_nll)
    

def calculate_metrics(y_true, y_pred, penalty_underestimate, model_name, df=None):
    custom_evaluator = CustomLoss(penalty_underestimate)
    cost = custom_evaluator.loss(y_true, y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    diff = y_pred - y_true
    size = len(y_true)

    metrics_dict = {
        "Model": model_name,
        "Cost": cost,
        "Standard MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "Underestimated portion": np.sum(diff < 0) / size,
        "Overestimated portion": np.sum(diff > 0) / size
    }

    # If no DataFrame exists, create one
    if df is None:
        return pd.DataFrame([metrics_dict])
    
    new_row = pd.DataFrame([metrics_dict])
    return pd.concat([df, new_row], ignore_index=True)