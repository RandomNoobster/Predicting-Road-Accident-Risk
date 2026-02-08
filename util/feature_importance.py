import shap
import matplotlib.pyplot as plt


def explain_model_with_shap(
    model,
    X,
    model_type="linear",
    show_beeswarm=True,
    show_waterfall=True,
    show_bar=False,
    show_heatmap=False,
    show_scatter=False,
    scatter_feature=None,
    waterfall_index=0,
):
    """
    Inputs:
        model: The trained model object.
        X: The input features (dataframe).
        model_type: "linear", "tree", or "deep".
        scatter_feature: String name of the feature to plot for scatter.
        waterfall_index: Index of the observation for the waterfall plot.
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

    if show_beeswarm:
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"Summary Plot ({model_type} model)")
        plt.tight_layout()
        plt.show()

    if show_waterfall:
        plt.figure(figsize=(8, 6))
        # Default index is 0, preserving old behavior
        shap.plots.waterfall(shap_values[waterfall_index], show=False)
        plt.title(f"Waterfall Plot (Obs: {waterfall_index})")
        plt.tight_layout()
        plt.show()

    if show_bar:
        plt.figure(figsize=(8, 6))
        shap.plots.bar(shap_values, show=False)
        plt.title("Global Feature Importance")
        plt.tight_layout()
        plt.show()

    if show_heatmap:
        plt.figure(figsize=(8, 6))
        shap.plots.heatmap(shap_values, show=False)
        plt.title("Feature Impact Heatmap")
        plt.tight_layout()
        plt.show()

    if show_scatter:
        plt.figure(figsize=(8, 6))
        if scatter_feature:
            shap.plots.scatter(shap_values[:, scatter_feature], show=False)
        else:
            shap.plots.scatter(shap_values, show=False)
        plt.title("Dependence Scatter Plot")
        plt.tight_layout()
        plt.show()

