# src/explainability.py

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path):
    return joblib.load(model_path)


def feature_importance(model, X, output_path):
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    top10 = importance_df.head(10)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=top10,
        x="importance",
        y="feature",
        palette="viridis"
    )

    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path)

    return importance_df


def generate_shap(model, X):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values


def shap_summary(shap_values, X, output_path):

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.tight_layout()
    plt.savefig(output_path)


def shap_bar(shap_values, X, output_path):

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()
    plt.savefig(output_path)


def save_force_plot(
    explainer,
    shap_values,
    X,
    idx,
    output_file
):

    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X.iloc[idx]
    )

    shap.save_html(
        output_file,
        force_plot
    )


def get_prediction_examples(
    X,
    y_true,
    y_pred
):

    results = X.copy()

    results["actual"] = y_true
    results["predicted"] = y_pred

    tp = results[
        (results.actual == 1) &
        (results.predicted == 1)
    ].index[0]

    fp = results[
        (results.actual == 0) &
        (results.predicted == 1)
    ].index[0]

    fn = results[
        (results.actual == 1) &
        (results.predicted == 0)
    ].index[0]

    return tp, fp, fn