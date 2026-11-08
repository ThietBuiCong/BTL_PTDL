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
    ID_COLUMNS,
    build_feature_frame,
    build_model_estimators,
    build_preprocessor,
    model_file_path,
    target_to_binary,
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
    }
    (MODELS_DIR / "model_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nMô hình có ROC-AUC cao nhất trên tập kiểm tra: {best_model}")
    print(pd.DataFrame(results).T.round(3).to_string())


if __name__ == "__main__":
    main()
