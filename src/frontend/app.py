import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from modeling import (
    ID_COLUMNS,
    ID_MAPPINGS,
    INPUT_COLUMNS,
    MEDICATION_COLUMNS,
    MODEL_NAMES,
    NUMERIC_COLUMNS,
    build_feature_frame,
    model_file_path,
    predict_all,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "code" / "models"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

RACE_OPTIONS = (
    "Caucasian",
    "AfricanAmerican",
    "Hispanic",
    "Asian",
    "Other",
    "Missing",
)
AGE_OPTIONS = (
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
)
GENDER_OPTIONS = ("Female", "Male", "Unknown/Invalid")
MEDICATION_OPTIONS = ("No", "Steady", "Up", "Down")
MEDICATION_LABELS = {
    "No": "Không dùng",
    "Steady": "Liều ổn định",
    "Up": "Tăng liều",
    "Down": "Giảm liều",
}
TEST_OPTIONS = ("Missing", "Norm", ">200", ">300")
A1C_OPTIONS = ("Missing", "Norm", ">7", ">8")
DEFAULTS = {
    "weight": "Missing",
    "time_in_hospital": 4,
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 5,
}


@st.cache_resource
def load_models() -> dict:
    missing = [
        model_file_path(MODELS_DIR, name)
        for name in MODEL_NAMES
        if not model_file_path(MODELS_DIR, name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Chưa có đủ bốn mô hình. Hãy chạy "
            "`python src\\frontend\\train_models.py` để huấn luyện."
        )
    return {
        model_name: joblib.load(model_file_path(MODELS_DIR, model_name))
        for model_name in MODEL_NAMES
    }


@st.cache_data
def load_model_metrics() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def format_model_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_NAMES:
        prediction = int(predictions[f"{model_name} - Dự đoán"].iloc[0])
        probability = float(
            predictions[f"{model_name} - Xác suất tái nhập viện"].iloc[0]
        )
        rows.append(
            {
                "Mô hình": model_name,
                "Dự đoán": (
                    "Có khả năng tái nhập viện"
                    if prediction
                    else "Không dự đoán tái nhập viện"
                ),
                "Xác suất tái nhập viện": probability,
            }
        )
    return pd.DataFrame(rows)


def best_model_summary(metrics: dict) -> str:
    name = metrics.get("best_model_by_roc_auc")
    score = metrics.get("metrics", {}).get(name, {}).get("ROC-AUC")
    if name is None or score is None:
        return "ROC-AUC cao nhất chưa có trong tệp đánh giá mô hình."
    return f"Mô hình có ROC-AUC cao nhất trên tập kiểm tra: {name} ({score:.3f})."


def _render_numeric_inputs(columns: tuple[str, ...], record: dict) -> None:
    numeric_labels = {
        "time_in_hospital": "Số ngày nằm viện",
        "num_lab_procedures": "Số xét nghiệm",
        "num_procedures": "Số thủ thuật",
        "num_medications": "Số loại thuốc",
        "number_outpatient": "Số lần khám ngoại trú trước đây",
        "number_emergency": "Số lần cấp cứu trước đây",
        "number_inpatient": "Số lần nhập viện nội trú trước đây",
        "number_diagnoses": "Số chẩn đoán",
    }
    columns_ui = st.columns(2)
    for index, column in enumerate(columns):
        record[column] = columns_ui[index % 2].number_input(
            numeric_labels[column],
            min_value=0,
            max_value=1000,
            value=DEFAULTS[column],
            key=f"patient_{column}",
        )


def render_single_prediction(models: dict, metrics: dict) -> None:
    with st.form("patient_input_form"):
        st.subheader("Thông tin nhân khẩu học")
        first, second = st.columns(2)
        record = {}
        record["race"] = first.selectbox(
            "Chủng tộc",
            RACE_OPTIONS,
            index=0,
            key="patient_race",
        )
        record["gender"] = second.selectbox(
            "Giới tính",
            GENDER_OPTIONS,
            key="patient_gender",
        )
        record["age"] = first.selectbox(
            "Nhóm tuổi",
            AGE_OPTIONS,
            index=6,
            key="patient_age",
        )
        record["weight"] = second.text_input(
            "Cân nặng (nhóm trong dữ liệu, ví dụ 75-100)",
            value=DEFAULTS["weight"],
            key="patient_weight",
        )

        st.subheader("Thông tin nhập viện")
        for column, label in (
            ("admission_type_id", "Hình thức nhập viện"),
            ("discharge_disposition_id", "Nơi chuyển đến khi xuất viện"),
            ("admission_source_id", "Nguồn tiếp nhận"),
        ):
            options = list(ID_MAPPINGS[column].items())
            selected = st.selectbox(
                label,
                options,
                format_func=lambda item: f"{item[0]} — {item[1]}",
                key=f"patient_{column}",
            )
            record[column] = selected[1]

        with st.expander("Lưu viện và dịch vụ"):
            _render_numeric_inputs(
                (
                    "time_in_hospital",
                    "num_lab_procedures",
                    "num_procedures",
                    "num_medications",
                    "number_outpatient",
                    "number_emergency",
                    "number_inpatient",
                    "number_diagnoses",
                ),
                record,
            )
            first, second = st.columns(2)
            record["payer_code"] = first.text_input(
                "Mã bên chi trả", value="Missing", key="patient_payer_code"
            )
            record["medical_specialty"] = second.text_input(
                "Chuyên khoa", value="Missing", key="patient_medical_specialty"
            )

        with st.expander("Chẩn đoán và xét nghiệm"):
            first, second = st.columns(2)
            record["diag_1"] = first.text_input(
                "Mã chẩn đoán chính (ICD-9)", value="428", key="patient_diag_1"
            )
            record["diag_2"] = second.text_input(
                "Mã chẩn đoán thứ hai (ICD-9)", value="250", key="patient_diag_2"
            )
            record["diag_3"] = first.text_input(
                "Mã chẩn đoán thứ ba (ICD-9)", value="401", key="patient_diag_3"
            )
            record["number_diagnoses"] = record.get(
                "number_diagnoses", DEFAULTS["number_diagnoses"]
            )
            record["max_glu_serum"] = second.selectbox(
                "Kết quả glucose huyết thanh",
                TEST_OPTIONS,
                key="patient_max_glu_serum",
            )
            record["A1Cresult"] = first.selectbox(
                "Kết quả A1C",
                A1C_OPTIONS,
                key="patient_A1Cresult",
            )

        with st.expander("Thuốc tiểu đường"):
            medication_ui = st.columns(3)
            for index, column in enumerate(MEDICATION_COLUMNS):
                record[column] = medication_ui[index % 3].selectbox(
                    column,
                    MEDICATION_OPTIONS,
                    format_func=lambda value: MEDICATION_LABELS[value],
                    key=f"patient_{column}",
                )
            first, second = st.columns(2)
            record["change"] = first.selectbox(
                "Thay đổi thuốc trong đợt này",
                ("No", "Ch"),
                format_func=lambda value: (
                    "Không thay đổi" if value == "No" else "Có thay đổi"
                ),
                key="patient_change",
            )
            record["diabetesMed"] = second.selectbox(
                "Có dùng thuốc tiểu đường",
                ("Yes", "No"),
                format_func=lambda value: "Có" if value == "Yes" else "Không",
                key="patient_diabetesMed",
            )

        submitted = st.form_submit_button("Dự đoán bằng 4 mô hình", type="primary")

    if not submitted:
        return

    try:
        features = build_feature_frame(pd.DataFrame([record], columns=INPUT_COLUMNS))
        predictions = predict_all(features, models)
    except (ValueError, KeyError) as error:
        st.error(f"Không thể thực hiện dự đoán: {error}")
        return

    st.subheader("Kết quả của bốn mô hình")
    st.dataframe(
        format_model_results(predictions).style.format(
            {"Xác suất tái nhập viện": "{:.1%}"}
        ),
        width="stretch",
        hide_index=True,
    )
    conclusion = predictions["Kết luận tổng hợp"].iloc[0]
    positive_votes = int(predictions["Số mô hình dự đoán tái nhập viện"].iloc[0])
    average_probability = float(predictions["Xác suất trung bình"].iloc[0])
    first, second = st.columns(2)
    first.metric("Kết luận tổng hợp", conclusion)
    second.metric("Xác suất trung bình", f"{average_probability:.1%}")
    st.caption(
        f"{positive_votes}/4 mô hình dự đoán có tái nhập viện. "
        + best_model_summary(metrics)
    )


def render_batch_prediction(models: dict, metrics: dict) -> None:
    st.subheader("Dự đoán hàng loạt từ CSV")
    st.write(
        "CSV cần có đầy đủ các thuộc tính đầu vào của bộ dữ liệu gốc. "
        "Có thể tải mẫu để xem tên cột và cách ghi giá trị."
    )
    template = pd.DataFrame([{column: "" for column in INPUT_COLUMNS}])
    template.loc[0, "age"] = "[70-80)"
    template.loc[0, "admission_type_id"] = 1
    template.loc[0, "discharge_disposition_id"] = 1
    template.loc[0, "admission_source_id"] = 7
    template.loc[0, "time_in_hospital"] = 4
    template.loc[0, "num_lab_procedures"] = 44
    template.loc[0, "num_procedures"] = 1
    template.loc[0, "num_medications"] = 15
    template.loc[0, "number_diagnoses"] = 5
    for column in ("number_outpatient", "number_emergency", "number_inpatient"):
        template.loc[0, column] = 0
    st.download_button(
        "Tải CSV mẫu",
        template.to_csv(index=False).encode("utf-8"),
        file_name="mau_du_doan_readmission.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader(
        "Chọn tệp CSV cần dự đoán",
        type=["csv"],
        help=(
            "Có thể giữ encounter_id, patient_nbr và readmitted trong tệp gốc; "
            "các cột này không được dùng làm đầu vào."
        ),
    )
    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()
    file_key = hashlib.sha256(file_bytes).hexdigest()
    try:
        input_data = pd.read_csv(uploaded_file, low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"Không đọc được CSV: {error}")
        return

    st.caption(f"Đã tải {len(input_data):,} dòng.")
    if "readmitted" in input_data.columns:
        st.info("Cột readmitted có trong tệp sẽ bị bỏ qua khi dự đoán.")

    if st.button("Dự đoán toàn bộ tệp", type="primary", key="predict_csv"):
        try:
            features = build_feature_frame(input_data)
            predictions = predict_all(features, models)
            input_without_target = input_data.drop(
                columns=["readmitted"], errors="ignore"
            ).reset_index(drop=True)
            results = pd.concat(
                [input_without_target, predictions.reset_index(drop=True)],
                axis=1,
            )
            results.insert(0, "Mã dòng", range(1, len(results) + 1))
            st.session_state["bulk_file_key"] = file_key
            st.session_state["bulk_results"] = results
        except (ValueError, KeyError) as error:
            st.error(f"Không thể dự đoán CSV: {error}")
            return

    if st.session_state.get("bulk_file_key") != file_key:
        return
    results = st.session_state.get("bulk_results")
    if results is None:
        return

    st.subheader("Kết quả dự đoán")
    positive = results["Kết luận tổng hợp"].eq(
        "Đa số dự đoán có tái nhập viện"
    ).sum()
    negative = results["Kết luận tổng hợp"].eq(
        "Đa số dự đoán không tái nhập viện"
    ).sum()
    split = results["Kết luận tổng hợp"].eq("Không đồng thuận (2–2)").sum()
    first, second, third = st.columns(3)
    first.metric("Đa số dự đoán tái nhập viện", f"{positive:,}")
    second.metric("Đa số dự đoán không tái nhập viện", f"{negative:,}")
    # third.metric("Không đồng thuận", f"{split:,}")
    st.caption(best_model_summary(metrics))
    st.dataframe(results.head(100), width="stretch", hide_index=True)
    st.download_button(
        "Tải kết quả CSV",
        results.to_csv(index=False).encode("utf-8-sig"),
        file_name="ket_qua_du_doan_readmission.csv",
        mime="text/csv",
    )


st.set_page_config(page_title="Dự đoán tái nhập viện", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f5f7fa;
        border: 1px solid #e6eaf0;
        padding: 1rem;
        border-radius: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Dự đoán tái nhập viện")
st.write(
    "Nhập đầy đủ thông tin bệnh nhân hoặc tải CSV để nhận dự đoán từ "
    "Logistic Regression, Random Forest, XGBoost và LightGBM."
)
st.caption(
    "Kết quả hỗ trợ phân tích dữ liệu, không thay thế đánh giá y tế. "
    "Mã loại nhập viện, xuất viện và nguồn tiếp nhận được chuẩn hóa theo "
    "IDS_mapping.csv."
)

try:
    models = load_models()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

metrics = load_model_metrics()
individual_tab, batch_tab = st.tabs(["Một bệnh nhân", "Tải CSV hàng loạt"])
with individual_tab:
    render_single_prediction(models, metrics)
with batch_tab:
    render_batch_prediction(models, metrics)
