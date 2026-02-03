import numpy as np

def custom_loss(y_true, y_pred, penalty_underestimate=5.0):
    diff = y_pred - y_true
    return np.mean(diff)