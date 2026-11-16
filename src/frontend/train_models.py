import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from modeling import (
    INPUT_COLUMNS,
    RETURN_TIMING_MODEL_NAME,
    build_feature_frame,
    build_model_estimators,
    build_preprocessor,
    build_return_timing_estimator,
    model_file_path,
    target_to_binary,
    target_to_return_timing,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "src" / "data" / "diabetic_data.csv"
MODELS_DIR = ROOT_DIR / "code" / "models"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu huấn luyện: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, low_memory=False)
    if "readmitted" not in data.columns:
        raise ValueError("Dữ liệu huấn luyện phải có cột readmitted.")

    features = build_feature_frame(data)
    target = target_to_binary(data["readmitted"])
    if target.nunique() != 2:
        raise ValueError("Cột readmitted phải có cả hai lớp NO và tái nhập viện.")

    groups = data["patient_nbr"] if "patient_nbr" in data.columns else None
    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            stratify=target,
            random_state=42,
        )
    else:
        train_indices, test_indices = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
                features, target, groups
            )
        )
        X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
        y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name, estimator in build_model_estimators().items():
        pipeline = Pipeline(
            [
                ("preprocessor", clone(build_preprocessor())),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        results[model_name] = {
            "Balanced accuracy": float(
                balanced_accuracy_score(y_test, predicted)
            ),
            "Precision": float(
                precision_score(y_test, predicted, zero_division=0)
            ),
            "Recall": float(recall_score(y_test, predicted, zero_division=0)),
            "F1": float(f1_score(y_test, predicted, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probability)),
        }
        joblib.dump(pipeline, model_file_path(MODELS_DIR, model_name))
        print(f"Đã huấn luyện và lưu: {model_name}")

    readmitted_mask = data["readmitted"].astype(str).str.strip().isin(
        {"<30", ">30"}
    )
    timing_features = features.loc[readmitted_mask]
    timing_target = target_to_return_timing(
        data.loc[readmitted_mask, "readmitted"]
    )
    timing_groups = groups.loc[readmitted_mask] if groups is not None else None
    if timing_target.nunique() != 2:
        raise ValueError(
            "Cột readmitted phải có cả hai lớp <30 và >30 để huấn luyện "
            "mô hình thời điểm."
        )

    if timing_groups is None:
        X_timing_train, X_timing_test, y_timing_train, y_timing_test = (
            train_test_split(
                timing_features,
                timing_target,
                test_size=0.2,
                stratify=timing_target,
                random_state=42,
            )
        )
    else:
        timing_train_indices, timing_test_indices = next(
            GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=42
            ).split(timing_features, timing_target, timing_groups)
        )
        X_timing_train = timing_features.iloc[timing_train_indices]
        X_timing_test = timing_features.iloc[timing_test_indices]
        y_timing_train = timing_target.iloc[timing_train_indices]
        y_timing_test = timing_target.iloc[timing_test_indices]

    if y_timing_test.nunique() != 2:
        raise ValueError(
            "Tập kiểm tra mô hình thời điểm phải có cả hai lớp <30 và >30."
        )

    timing_pipeline = Pipeline(
        [
            ("preprocessor", clone(build_preprocessor())),
            ("classifier", build_return_timing_estimator()),
        ]
    )
    timing_pipeline.fit(X_timing_train, y_timing_train)
    timing_predicted = timing_pipeline.predict(X_timing_test)
    timing_probability = timing_pipeline.predict_proba(X_timing_test)[:, 1]
    timing_metrics = {
        "Balanced accuracy": float(
            balanced_accuracy_score(y_timing_test, timing_predicted)
        ),
        "Precision": float(
            precision_score(y_timing_test, timing_predicted, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_timing_test, timing_predicted, zero_division=0)
        ),
        "F1": float(f1_score(y_timing_test, timing_predicted, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_timing_test, timing_probability)),
    }
    joblib.dump(
        timing_pipeline,
        model_file_path(MODELS_DIR, RETURN_TIMING_MODEL_NAME),
    )
    print(f"Đã huấn luyện và lưu: {RETURN_TIMING_MODEL_NAME}")

    best_model = max(results, key=lambda name: results[name]["ROC-AUC"])
    metadata = {
        "feature_columns": list(INPUT_COLUMNS),
        "id_columns_mapped_with": "src/data/IDS_mapping.csv",
        "training_rows": int(len(data)),
        "test_rows": int(len(X_test)),
        "patient_group_split": groups is not None,
        "sklearn_version": sklearn.__version__,
        "best_model_by_roc_auc": best_model,
        "metrics": results,
        "return_timing_model": {
            "positive_class": "<30",
            "negative_class": ">30",
            "training_rows": int(len(timing_features)),
            "test_rows": int(len(X_timing_test)),
            "patient_group_split": timing_groups is not None,
            "metrics": timing_metrics,
        },
    }
    (MODELS_DIR / "model_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nMô hình có ROC-AUC cao nhất trên tập kiểm tra: {best_model}")
    print(pd.DataFrame(results).T.round(3).to_string())
    print("\nĐánh giá mô hình thời điểm tái nhập viện:")
    print(pd.Series(timing_metrics).round(3).to_string())


if __name__ == "__main__":
    main()
