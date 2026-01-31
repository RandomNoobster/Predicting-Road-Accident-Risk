import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# From "Methods and Tools":
# We wrap data loading to ensure consistency across the team's experiments.
def load_data(path):
    return pd.read_csv(path)

# From "Anomaly Detection" (Slide: A Simple Cost Model):
# We evaluate the model based on business value, not just accuracy.
# This implements the logic: cost = c_alrm * FP + c_missed * FN
def calculate_cost(y_true, y_pred_prob, threshold, c_fp, c_fn):
    """
    Calculates business cost based on the Anomaly Detection lecture logic.
    c_fp: Cost of False Positive (False Alarm / Wasted Intervention)
    c_fn: Cost of False Negative (Missed Accident)
    """
    # Convert probabilities to binary decisions based on the threshold
    preds = (y_pred_prob >= threshold).astype(int)
    
    # Extract confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    # Compute total cost (we assume TN and TP have 0 cost for this specific model)
    total_cost = (fp * c_fp) + (fn * c_fn)
    return total_cost

# From "Anomaly Detection" (Slide: Threshold Optimization):
# We use a simple line search to find the optimal operating point (epsilon/theta).
def opt_threshold(y_true, y_pred_prob, c_fp, c_fn):
    """
    Line search for threshold optimization.
    """
    # Define a range of "sampled" thresholds (as seen in the lecture)
    thresholds = np.linspace(0, 1, 101)
    
    # Evaluate cost for each threshold
    costs = [calculate_cost(y_true, y_pred_prob, t, c_fp, c_fn) for t in thresholds]
    
    # Pick the best one (argmin)
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]