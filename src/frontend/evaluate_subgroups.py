import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from modeling import build_feature_frame, model_file_path, target_to_binary


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "src" / "data" / "diabetic_data.csv"
MODELS_DIR = ROOT_DIR / "code" / "models"
OUTPUT_PATH = ROOT_DIR / "code" / "subgroup_metrics.csv"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
GROUP_COLUMNS = ("race", "gender", "age")
BOOTSTRAP_REPLICATES = 2000
RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.5


def _bootstrap_intervals(
    target: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    patient_ids: np.ndarray,
    replicates: int,
    random_state: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    cluster_rows = list(
        pd.Series(np.arange(len(target)))
        .groupby(patient_ids, sort=False)
        .indices.values()
    )
    rng = np.random.default_rng(random_state)
    auc_values = []
    recall_values = []

    for _ in range(replicates):
        selected_clusters = rng.integers(
            0, len(cluster_rows), size=len(cluster_rows)
        )
        rows = np.concatenate(
            [cluster_rows[index] for index in selected_clusters]
        )
        sampled_target = target[rows]
        sampled_positive = sampled_target == 1
        if sampled_positive.any() and (~sampled_positive).any():
            auc_values.append(
                float(roc_auc_score(sampled_target, probabilities[rows]))
            )
        if sampled_positive.any():
            recall_values.append(
                float(predictions[rows][sampled_positive].mean())
            )

    auc_interval = (
        tuple(np.percentile(auc_values, [2.5, 97.5]))
        if auc_values
        else (float("nan"), float("nan"))
    )
    recall_interval = (
        tuple(np.percentile(recall_values, [2.5, 97.5]))
        if recall_values
        else (float("nan"), float("nan"))
    )
    return auc_interval, recall_interval


def calculate_subgroup_metrics(
    data: pd.DataFrame,
    target: pd.Series,
    probabilities: np.ndarray,
    patient_ids: pd.Series,
    replicates: int = BOOTSTRAP_REPLICATES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    if not (len(data) == len(target) == len(probabilities) == len(patient_ids)):
        raise ValueError(
            "Số dòng dữ liệu, nhãn, xác suất và ID bệnh nhân phải khớp."
        )
    if replicates < 1:
        raise ValueError("Số lần bootstrap phải lớn hơn 0.")

    predicted = (probabilities >= CLASSIFICATION_THRESHOLD).astype("int8")
    evaluation = pd.DataFrame(
        {
            "target": target.to_numpy(),
            "probability": probabilities,
            "prediction": predicted,
            "patient_nbr": patient_ids.to_numpy(),
        }
    )
    for column in GROUP_COLUMNS:
        values = data[column].astype("string").fillna("Missing")
        evaluation[column] = values.replace("?", "Missing").to_numpy()

    rows: list[dict[str, Any]] = []
    for column_index, column in enumerate(GROUP_COLUMNS):
        for subgroup, positions in evaluation.groupby(
            column, sort=False, dropna=False
        ).indices.items():
            subgroup_data = evaluation.iloc[positions]
            subgroup_target = subgroup_data["target"].to_numpy(dtype="int8")
            subgroup_probabilities = subgroup_data["probability"].to_numpy()
            subgroup_predictions = subgroup_data["prediction"].to_numpy()
            subgroup_patients = subgroup_data["patient_nbr"].to_numpy()
            positive_count = int(subgroup_target.sum())
            negative_count = len(subgroup_target) - positive_count
            subgroup_seed = random_state + column_index * 1000 + len(rows)

            if positive_count and negative_count:
                auc = float(
                    roc_auc_score(subgroup_target, subgroup_probabilities)
                )
            else:
                auc = float("nan")
            recall = (
                float(subgroup_predictions[subgroup_target == 1].mean())
                if positive_count
                else float("nan")
            )
            auc_interval, recall_interval = _bootstrap_intervals(
                subgroup_target,
                subgroup_probabilities,
                subgroup_predictions,
                subgroup_patients,
                replicates,
                subgroup_seed,
            )
            rows.append(
                {
                    "attribute": column,
                    "group": str(subgroup),
                    "n": len(subgroup_target),
                    "positive_n": positive_count,
                    "auc": auc,
                    "auc_ci_low": auc_interval[0],
                    "auc_ci_high": auc_interval[1],
                    "recall": recall,
                    "recall_ci_low": recall_interval[0],
                    "recall_ci_high": recall_interval[1],
                    "bootstrap_replicates": replicates,
                    "unique_patients": int(
                        pd.Series(subgroup_patients).nunique()
                    ),
                    "threshold": CLASSIFICATION_THRESHOLD,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_PATH}")
    model_path = model_file_path(MODELS_DIR, "LightGBM")
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình LightGBM: {model_path}")

    data = pd.read_csv(DATA_PATH, low_memory=False)
    if "patient_nbr" not in data:
        raise ValueError("Dữ liệu cần có patient_nbr để chia theo bệnh nhân.")
    target = target_to_binary(data["readmitted"])
    _, test_indices = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=RANDOM_STATE
        ).split(data, target, data["patient_nbr"])
    )
    test_data = data.iloc[test_indices]
    test_target = target.iloc[test_indices]
    features = build_feature_frame(test_data)
    model = joblib.load(model_path)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(features)[:, positive_index]

    results = calculate_subgroup_metrics(
        test_data,
        test_target,
        probabilities,
        test_data["patient_nbr"],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False, float_format="%.8f")

    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(
        f"Mô hình: LightGBM (ROC-AUC holdout "
        f"{metadata['metrics']['LightGBM']['ROC-AUC']:.3f}); "
        f"ngưỡng Recall: {CLASSIFICATION_THRESHOLD:.2f}"
    )
    print(
        f"Holdout: {len(test_data):,} lượt khám, "
        f"{test_data['patient_nbr'].nunique():,} bệnh nhân; "
        f"bootstrap theo cụm bệnh nhân, {BOOTSTRAP_REPLICATES:,} lần."
    )
    print(
        results.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"\nĐã lưu kết quả: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
