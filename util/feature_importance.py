import shap
import matplotlib.pyplot as plt


def explain_model_with_shap(model, X_test, model_type="linear"):
    """
    Inputs:
        model: The trained model object.
        X_data: The input features (dataframe).
        model_type: "linear" for regression/logistic, "tree" for random forest/xgboost.
    """

    if model_type == "linear":
        explainer = shap.LinearExplainer(model, X_test)
    elif model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_test)

    shap_values = explainer(X_test)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"Summary Plot ({model_type} model)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"Waterfall Plot ({model_type} model)")
    plt.tight_layout()
    plt.show()

