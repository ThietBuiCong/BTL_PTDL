import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "src" / "data"

MODEL_NAMES = (
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
)

ID_COLUMNS = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)
NUMERIC_COLUMNS = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)
MEDICATION_COLUMNS = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)
INPUT_COLUMNS = (
    "race",
    "gender",
    "age",
    "weight",
    *ID_COLUMNS,
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    *MEDICATION_COLUMNS,
    "change",
    "diabetesMed",
)
CATEGORICAL_COLUMNS = tuple(
    column for column in INPUT_COLUMNS if column not in NUMERIC_COLUMNS
)


def load_id_mappings(path: Path | None = None) -> dict[str, dict[str, str]]:
    mapping_path = path or DATA_DIR / "IDS_mapping.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy bảng mã: {mapping_path}")

    mappings: dict[str, dict[str, str]] = {}
    section: str | None = None
    with mapping_path.open(encoding="utf-8-sig", newline="") as mapping_file:
        for row in csv.reader(mapping_file):
            if not row:
                continue
            key = row[0].strip()
            description = row[1].strip() if len(row) > 1 else ""
            if key in ID_COLUMNS and description.lower() == "description":
                section = key
                mappings[section] = {}
                continue
            if section is None or not key or not description:
                continue
            if key.isdigit():
                normalized_description = (
                    "Unknown/Invalid"
                    if description.upper() == "NULL"
                    else description
                )
                mappings[section][key] = normalized_description

    missing_mappings = set(ID_COLUMNS) - set(mappings)
    if missing_mappings:
        raise ValueError(
            "Bảng IDS_mapping thiếu nhóm mã: "
            + ", ".join(sorted(missing_mappings))
        )
    return mappings


ID_MAPPINGS = load_id_mappings()
ID_DESCRIPTIONS = {
    column: {
        description.casefold(): description
        for description in mapping.values()
    }
    for column, mapping in ID_MAPPINGS.items()
}


def _normalize_id_value(value: object, column: str) -> str:
    if pd.isna(value):
        return "Unknown/Invalid"
    text = str(value).strip()
    if not text or text == "?" or text.casefold() == "null":
        return "Unknown/Invalid"

    try:
        numeric_id = str(int(float(text)))
    except ValueError:
        numeric_id = ""
    if numeric_id in ID_MAPPINGS[column]:
        return ID_MAPPINGS[column][numeric_id]

    description = ID_DESCRIPTIONS[column].get(text.casefold())
    if description is None:
        raise ValueError(
            f"Giá trị {text!r} không có trong IDS_mapping cho cột {column}."
        )
    return description


def build_feature_frame(records: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(INPUT_COLUMNS) - set(records.columns))
    if missing_columns:
        raise ValueError(
            "CSV thiếu các thuộc tính bắt buộc: " + ", ".join(missing_columns)
        )

    features = records.loc[:, INPUT_COLUMNS].copy()
    for column in ID_COLUMNS:
        features[column] = features[column].map(
            lambda value, name=column: _normalize_id_value(value, name)
        )

    for column in NUMERIC_COLUMNS:
        original_values = features[column].replace("?", np.nan)
        values = pd.to_numeric(original_values, errors="coerce")
        invalid = original_values.notna() & values.isna()
        if invalid.any():
            invalid_values = sorted(original_values[invalid].astype(str).unique())
            raise ValueError(
                f"Cột {column} có giá trị không hợp lệ: "
                + ", ".join(invalid_values)
            )
        if (values.dropna() < 0).any():
            raise ValueError(f"Cột {column} không được chứa số âm.")
        features[column] = values

    for column in CATEGORICAL_COLUMNS:
        values = features[column].fillna("").astype(str).str.strip()
        values = values.replace({"?": "Missing", "": "Missing"})
        if column in {"change", "diabetesMed"}:
            values = values.str.lower().replace(
                {"no": "No", "ch": "Ch", "yes": "Yes"}
            )
            invalid = ~values.isin({"No", "Ch", "Yes", "Missing"})
            if invalid.any():
                raise ValueError(
                    f"Cột {column} nhận giá trị No, Ch, Yes hoặc để trống."
                )
        features[column] = values

    return features.loc[:, INPUT_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_COLUMNS)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
    )


def build_model_estimators() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=9,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            verbosity=-1,
            random_state=42,
        ),
    }


def target_to_binary(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    return (~normalized.isin({"no", "0", "false"})).astype("int8")


def model_file_path(models_dir: Path, model_name: str) -> Path:
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    return models_dir / filename


def predict_all(features: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    predictions: dict[str, object] = {}
    for model_name in MODEL_NAMES:
        model = models[model_name]
        predicted_class = model.predict(features).astype(int)
        positive_class_index = list(model.classes_).index(1)
        probability = model.predict_proba(features)[:, positive_class_index]
        predictions[f"{model_name} - Dự đoán"] = predicted_class
        predictions[f"{model_name} - Xác suất tái nhập viện"] = probability

    prediction_frame = pd.DataFrame(predictions, index=features.index)
    class_columns = [f"{name} - Dự đoán" for name in MODEL_NAMES]
    probability_columns = [
        f"{name} - Xác suất tái nhập viện" for name in MODEL_NAMES
    ]
    positive_votes = prediction_frame[class_columns].sum(axis=1)
    prediction_frame["Số mô hình dự đoán tái nhập viện"] = positive_votes
    prediction_frame["Xác suất trung bình"] = prediction_frame[
        probability_columns
    ].mean(axis=1)
    prediction_frame["Kết luận tổng hợp"] = "Không đồng thuận (2–2)"
    prediction_frame.loc[positive_votes > 2, "Kết luận tổng hợp"] = (
        "Đa số dự đoán có tái nhập viện"
    )
    prediction_frame.loc[positive_votes < 2, "Kết luận tổng hợp"] = (
        "Đa số dự đoán không tái nhập viện"
    )
    return prediction_frame
