import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


def load_data(path):
    """
    Standardizes data loading across the project.
    Ref: 'Methods and Tools' - Reproducibility.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    return pd.read_csv(path)


def plot_pred_scatter(y_true, y_pred, figsize=(10, 6), title='Prediction Accuracy'):
    """
    Scatter plot for regression evaluation.
    Ref: 'RUL Prediction as Regression' (Slide: Baseline Evaluation).
    
    Visualizes how well predictions match ground truth.
    The red dashed line represents perfect prediction (y = x).
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Plot diagonal line for reference
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Risk')
    plt.ylabel('Predicted Risk')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.title(title)
    plt.show()

def plot_importance(importance_series, title='Feature Importance', top_n=20, figsize=(10, 8)):
    """
    Bar chart for feature importance.
    Ref: 'Non-Linear Models' (Slide: Important Attributes).
    
    Handles sorting and limiting the number of features displayed 
    to focus on the most relevant correlates.
    """
    plt.figure(figsize=figsize)
    # Plot top N features sorted by absolute value (if signed) or magnitude
    importance_series.abs().sort_values(ascending=True).tail(top_n).plot(kind='barh')
    
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(title)
    plt.grid(True, axis='x', alpha=0.5)
    plt.show()


def asymmetric_risk_loss(y_true, y_pred, penalty_underestimate=5.0):
    """
    Custom 'Industrial' metric.
    Ref: 'RUL Prediction' (Slide: Cost Model).
    
    In safety-critical applications (like road accidents or equipment failure),
    underestimating risk is often much more costly than overestimating it.
    
    Args:
        penalty_underestimate: Multiplier for errors where Pred < True.
    """
    diff = y_pred - y_true
    
    # Logic: 
    # If diff < 0 (Underestimate), cost = |diff| * penalty
    # If diff > 0 (Overestimate), cost = |diff| * 1
    weights = np.where(diff < 0, penalty_underestimate, 1.0)
    
    weighted_mae = np.mean(np.abs(diff) * weights)
    return weighted_mae