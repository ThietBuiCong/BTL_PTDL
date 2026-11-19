from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import shap

from modeling import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _feature_groups(preprocessor: Any) -> list[str]:
    encoder = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["onehot"]
    )
    feature_groups = list(NUMERIC_COLUMNS)
    for column, categories in zip(
        CATEGORICAL_COLUMNS, encoder.categories_, strict=True
    ):
        feature_groups.extend([column] * len(categories))
    if len(feature_groups) != len(preprocessor.get_feature_names_out()):
        raise ValueError("Không thể ánh xạ SHAP về các đặc trưng gốc.")
    return feature_groups


def _positive_class_values(explanation: Any, class_index: int) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, class_index]
    if values.ndim != 2:
        raise ValueError("SHAP trả về cấu trúc đóng góp không hợp lệ.")
    return values


def explain_local_prediction(
    model: Any,
    features: pd.DataFrame,
    background: pd.DataFrame,
) -> tuple[pd.Series, float, float]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed_background = _to_dense(preprocessor.transform(background))
    transformed_features = _to_dense(preprocessor.transform(features))

    explainer = shap.TreeExplainer(
        classifier,
        data=transformed_background,
        model_output="probability",
        feature_perturbation="interventional",
    )
    explanation = explainer(transformed_features)
    class_index = list(classifier.classes_).index(1)
    shap_values = _positive_class_values(explanation, class_index)
    if shap_values.ndim != 2 or shap_values.shape[0] != 1:
        raise ValueError("SHAP chỉ hỗ trợ giải thích một bản ghi mỗi lần.")

    feature_groups = _feature_groups(preprocessor)
    if len(feature_groups) != shap_values.shape[1]:
        raise ValueError("Không thể ánh xạ đóng góp SHAP về các đặc trưng gốc.")

    contributions = pd.Series(
        shap_values[0], index=feature_groups, dtype="float64"
    ).groupby(level=0, sort=False).sum()
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    probability = float(
        classifier.predict_proba(transformed_features)[0][class_index]
    )
    if not np.isclose(
        base_value + contributions.sum(), probability, rtol=1e-4, atol=1e-5
    ):
        raise ValueError("Tổng đóng góp SHAP không khớp xác suất dự đoán.")

    return contributions, base_value, probability


def create_local_waterfall(
    contributions: pd.Series,
    base_value: float,
    features: pd.DataFrame,
    max_display: int = 12,
) -> bytes:
    import matplotlib.pyplot as plt

    if len(features) != 1:
        raise ValueError("Waterfall SHAP chỉ hỗ trợ một bệnh nhân mỗi lần.")
    explanation = shap.Explanation(
        values=contributions.to_numpy(),
        base_values=base_value,
        data=np.asarray(
            [str(features.iloc[0][name]) for name in contributions.index],
            dtype=object,
        ),
        feature_names=contributions.index.tolist(),
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    figure = plt.gcf()
    figure.set_size_inches(11, 7.5)
    figure.axes[0].set_xlabel(
        "SHAP value (impact on predicted probability)"
    )
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def explain_global_importance(
    model: Any,
    features: pd.DataFrame,
    background: pd.DataFrame,
) -> pd.Series:
    importance, _ = _global_shap_explanation(model, features, background)
    return importance


def explain_global_beeswarm(
    model: Any,
    features: pd.DataFrame,
    background: pd.DataFrame,
    max_display: int = 20,
) -> tuple[pd.Series, bytes, bytes, pd.DataFrame]:
    import matplotlib.pyplot as plt

    importance, explanation = _global_shap_explanation(
        model, features, background
    )
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    figure = plt.gcf()
    figure.set_size_inches(11, max(6, max_display * 0.35))
    figure.axes[0].set_xlabel(
        "SHAP value (impact on predicted probability)"
    )
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    beeswarm_image = buffer.getvalue()

    feature_index = explanation.feature_names.index("number_inpatient")
    inpatient_count = np.asarray(explanation.data[:, feature_index], dtype=float)
    shap_contribution = np.asarray(
        explanation.values[:, feature_index], dtype=float
    )
    dependence = pd.DataFrame(
        {
            "number_inpatient": inpatient_count,
            "shap": shap_contribution,
        }
    )
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, np.inf]
    labels = ["0", "1", "2", "3", "4-5", "6+"]
    dependence["Lượt nhập viện"] = pd.cut(
        dependence["number_inpatient"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    summary = (
        dependence.groupby("Lượt nhập viện", observed=False)["shap"]
        .agg(["size", "mean"])
        .rename(columns={"size": "Số bản ghi", "mean": "SHAP trung bình"})
        .reindex(labels)
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(
        inpatient_count,
        shap_contribution * 100,
        alpha=0.28,
        s=18,
        color="#3973ac",
        edgecolors="none",
        label="Bản ghi trong mẫu",
    )
    mean_by_count = (
        dependence.groupby("number_inpatient", observed=True)["shap"]
        .mean()
        .sort_index()
    )
    axis.plot(
        mean_by_count.index,
        mean_by_count.to_numpy() * 100,
        color="#d62728",
        marker="o",
        linewidth=2,
        markersize=4,
        label="SHAP trung bình tại mỗi số lượt",
    )
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Số lượt nhập viện nội trú trước đây (number_inpatient)")
    axis.set_ylabel("Đóng góp SHAP lên xác suất tái nhập viện (điểm %)")
    axis.set_title("Dependence plot SHAP: number_inpatient")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    dependence_image = buffer.getvalue()

    return importance, beeswarm_image, dependence_image, summary


def _global_shap_explanation(
    model: Any,
    features: pd.DataFrame,
    background: pd.DataFrame,
) -> tuple[pd.Series, shap.Explanation]:
    if features.empty:
        raise ValueError("Cần ít nhất một bản ghi để tính SHAP toàn cục.")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed_background = _to_dense(preprocessor.transform(background))
    transformed_features = _to_dense(preprocessor.transform(features))
    explainer = shap.TreeExplainer(
        classifier,
        data=transformed_background,
        model_output="probability",
        feature_perturbation="interventional",
    )
    explanation = explainer(transformed_features)
    class_index = list(classifier.classes_).index(1)
    shap_values = _positive_class_values(explanation, class_index)
    feature_groups = _feature_groups(preprocessor)
    if shap_values.shape != (len(features), len(feature_groups)):
        raise ValueError("Kích thước SHAP không khớp với dữ liệu đầu vào.")

    numeric_imputer = (
        preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    )
    numeric_values = numeric_imputer.transform(
        features.loc[:, NUMERIC_COLUMNS]
    )
    plot_values = np.concatenate(
        [numeric_values, transformed_features[:, len(NUMERIC_COLUMNS) :]],
        axis=1,
    )
    feature_names = [
        name.split("__", maxsplit=1)[-1]
        for name in preprocessor.get_feature_names_out()
    ]
    grouped_values = (
        pd.DataFrame(shap_values, columns=feature_groups)
        .T.groupby(level=0, sort=False)
        .sum()
        .T
    )
    importance = grouped_values.abs().mean().sort_values(ascending=False)
    explanation_data = shap.Explanation(
        values=shap_values,
        base_values=np.asarray(explanation.base_values).reshape(-1),
        data=plot_values,
        feature_names=feature_names,
    )
    return importance, explanation_data
