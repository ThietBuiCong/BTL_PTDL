from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "src" / "analysis" / "models"
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.set_page_config(
    page_title="Dự đoán tái nhập viện",
    page_icon="🏥",
    layout="centered",
)

st.title("Dự đoán nguy cơ tái nhập viện")
st.write("Nhập thông tin bệnh nhân và chọn mô hình để nhận dự đoán.")
st.caption(
    "Kết quả chỉ là dự đoán của mô hình trên dữ liệu đã huấn luyện, "
    "không thay thế đánh giá hoặc tư vấn của nhân viên y tế."
)

model_label = st.selectbox("Mô hình", options=list(MODEL_OPTIONS))

with st.form("patient_input_form"):
    st.subheader("Thông tin bệnh nhân")
    left, right = st.columns(2)

    with left:
        age = st.number_input("Tuổi", min_value=18, max_value=89, value=53)
        length_of_stay_days = st.number_input(
            "Số ngày nằm viện", min_value=1, max_value=29, value=15
        )
        num_diagnoses = st.number_input(
            "Số chẩn đoán", min_value=1, max_value=14, value=8
        )
        num_medications = st.number_input(
            "Số loại thuốc", min_value=1, max_value=19, value=10
        )
        prev_admissions = st.number_input(
            "Số lần nhập viện trước", min_value=0, max_value=9, value=4
        )

    with right:
        glucose_level = st.number_input(
            "Mức glucose", min_value=60.0, max_value=300.0, value=179.9, step=0.1
        )
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=32.6, step=0.1)
        has_diabetes = st.selectbox(
            "Có tiểu đường", options=[0, 1], format_func=lambda value: "Có" if value else "Không"
        )
        discharge_type = st.selectbox(
            "Mã loại xuất viện", options=[0, 1, 2]
        )

    submitted = st.form_submit_button("Dự đoán", type="primary")

if submitted:
    model_path = MODELS_DIR / MODEL_OPTIONS[model_label]
    if not model_path.is_file():
        st.error(f"Không tìm thấy model: {model_path}")
    else:
        try:
            model = load_model(str(model_path))
            patient = pd.DataFrame([{
                "age": age,
                "length_of_stay_days": length_of_stay_days,
                "num_diagnoses": num_diagnoses,
                "num_medications": num_medications,
                "prev_admissions": prev_admissions,
                "glucose_level": glucose_level,
                "bmi": bmi,
                "has_diabetes": has_diabetes,
                "discharge_type": discharge_type,
            }])
            prediction = int(model.predict(patient)[0])
            class_labels = list(model.classes_)
            positive_class_index = class_labels.index(1)
            probability = float(model.predict_proba(patient)[0][positive_class_index])

            if prediction == 1:
                st.warning("Mô hình dự đoán: Có khả năng tái nhập viện trong 30 ngày.")
            else:
                st.success("Mô hình dự đoán: Không tái nhập viện trong 30 ngày.")
            st.metric("Xác suất dự đoán tái nhập viện", f"{probability:.1%}")
        except (ValueError, KeyError, OSError) as error:
            st.error(f"Không thể thực hiện dự đoán: {error}")
