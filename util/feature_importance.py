import shap
import matplotlib.pyplot as plt


def explain_model_with_shap(model, X, model_type="linear"):
    """
    Inputs:
        model: The trained model object.
        X: The input features (dataframe).
        model_type: "linear" for regression/logistic, "tree" for random forest/xgboost.
    """

    if model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
    elif model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "deep":
        explainer = shap.DeepExplainer(model, X)
    else:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)

    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"Summary Plot ({model_type} model)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"Waterfall Plot ({model_type} model)")
    plt.tight_layout()
    plt.show()

