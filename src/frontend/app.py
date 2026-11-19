import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from explainability import explain_local_prediction
from modeling import (
    ID_MAPPINGS,
    INPUT_COLUMNS,
    MEDICATION_COLUMNS,
    MODEL_NAMES,
    RETURN_TIMING_MODEL_NAME,
    build_feature_frame,
    model_file_path,
    predict_all,
    predict_return_timing,
    target_to_binary,
)
from explainability import (
    create_local_waterfall,
    explain_global_beeswarm,
    explain_local_prediction,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "src" / "data" / "diabetic_data.csv"
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
GLOBAL_SHAP_SAMPLE_SIZE = 1000


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


@st.cache_resource
def load_return_timing_model():
    model_path = model_file_path(MODELS_DIR, RETURN_TIMING_MODEL_NAME)
    if not model_path.is_file():
        raise FileNotFoundError(
            "Chưa có mô hình dự đoán thời điểm tái nhập viện. Hãy chạy "
            "`python src\\frontend\\train_models.py` để huấn luyện."
        )
    return joblib.load(model_path)


@st.cache_data
def load_model_metrics() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_shap_background() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    target = target_to_binary(data["readmitted"])
    samples = []
    for label in (0, 1):
        class_rows = data.loc[target.eq(label)]
        if len(class_rows) < 50:
            raise ValueError(
                "Cần ít nhất 50 bản ghi cho mỗi lớp để tạo nền SHAP."
            )
        samples.append(class_rows.sample(n=50, random_state=42))
    return build_feature_frame(pd.concat(samples))


@st.cache_data
def load_shap_global_sample() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    sample = data.sample(
        n=min(GLOBAL_SHAP_SAMPLE_SIZE, len(data)), random_state=42
    )
    return build_feature_frame(sample)


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
    except (FileNotFoundError, ValueError, KeyError) as error:
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
    if conclusion == "Đa số dự đoán có tái nhập viện":
        try:
            timing = predict_return_timing(
                features, load_return_timing_model()
            ).iloc[0]
        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(f"Không thể dự đoán thời điểm tái nhập viện: {error}")
            return
        st.success(
            f"Mô hình thời điểm dự đoán: "
            f"{timing['Khoảng thời gian tái nhập viện']} "
            f"(xác suất trong 30 ngày: "
            f"{timing['Xác suất tái nhập viện trong 30 ngày']:.1%})."
        )
    elif conclusion == "Đa số dự đoán không tái nhập viện":
        st.info("Không dự đoán tái nhập viện; bỏ qua mô hình dự đoán thời điểm.")
    else:
        st.info(
            "Các mô hình chưa đồng thuận; bỏ qua mô hình dự đoán thời điểm."
        )

    first, second = st.columns(2)
    first.metric("Kết luận tổng hợp", conclusion)
    second.metric("Xác suất trung bình", f"{average_probability:.1%}")
    st.caption(
        f"{positive_votes}/4 mô hình dự đoán có tái nhập viện. "
        + best_model_summary(metrics)
    )

    try:
        contributions, baseline, probability = explain_local_prediction(
            models["LightGBM"], features, load_shap_background()
        )
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(f"Không thể tạo giải thích SHAP: {error}")
        return

    st.subheader("Giải thích cục bộ bằng SHAP (LightGBM)")
    st.caption(
        "SHAP phân rã xác suất tái nhập viện của bệnh nhân thành đóng góp "
        "của từng đặc trưng. Mỗi biến số có đơn vị điểm phần trăm; đặc trưng "
        "dương làm tăng, đặc trưng âm làm giảm xác suất so với mức nền."
    )
    try:
        waterfall = create_local_waterfall(
            contributions, baseline, features
        )
    except ValueError as error:
        st.error(f"Không thể tạo Waterfall SHAP: {error}")
        return
    st.markdown(
        "**Waterfall SHAP cho bệnh nhân nguy cơ cao**"
        if probability >= 0.5
        else "**Waterfall SHAP cho bệnh nhân này**"
    )
    st.caption(
        "Điểm bắt đầu là xác suất nền; các thanh đỏ/xanh lần lượt đẩy xác "
        "suất lên/xuống đến dự đoán của LightGBM. Mốc nguy cơ cao ở đây là "
        "LightGBM dự đoán lớp tái nhập viện (xác suất từ 50%)."
    )
    st.image(waterfall, width="stretch")
    baseline_column, prediction_column = st.columns(2)
    baseline_column.metric("Xác suất nền", f"{baseline:.1%}")
    prediction_column.metric("Xác suất dự đoán", f"{probability:.1%}")

    local_explanation = pd.DataFrame(
        {
            "Đặc trưng": contributions.index,
            "Giá trị bệnh nhân": [
                str(features.iloc[0][column])
                for column in contributions.index
            ],
            "Đóng góp (điểm %)": contributions.to_numpy() * 100,
        }
    )
    local_explanation["Hướng tác động"] = local_explanation[
        "Đóng góp (điểm %)"
    ].map(
        lambda value: (
            "Tăng nguy cơ"
            if value > 0
            else "Giảm nguy cơ"
            if value < 0
            else "Không đổi"
        )
    )
    local_explanation = local_explanation.reindex(
        local_explanation["Đóng góp (điểm %)"]
        .abs()
        .sort_values(ascending=False)
        .index
    )
    st.dataframe(
        local_explanation.style.format({"Đóng góp (điểm %)": "{:+.2f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Xác suất dự đoán = xác suất nền + tổng đóng góp SHAP. "
        "Đây là giải thích hành vi của mô hình, không khẳng định quan hệ nhân quả."
    )


def render_batch_prediction(models: dict, metrics: dict) -> None:
    st.subheader("Dự đoán hàng loạt từ CSV")
    st.write(
        "CSV cần có đầy đủ các thuộc tính đầu vào của bộ dữ liệu gốc. "
        "Có thể tải mẫu để xem tên cột và cách ghi giá trị."
    )
    template = pd.DataFrame(
        [{column: "" for column in INPUT_COLUMNS}], dtype=object
    )
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
            readmitted_mask = predictions["Kết luận tổng hợp"].eq(
                "Đa số dự đoán có tái nhập viện"
            )
            predictions = predictions.copy()
            predictions["Khoảng thời gian tái nhập viện"] = (
                "Không áp dụng"
            )
            predictions["Xác suất tái nhập viện trong 30 ngày"] = pd.Series(
                index=predictions.index, dtype="float64"
            )
            if readmitted_mask.any():
                timing = predict_return_timing(
                    features.loc[readmitted_mask],
                    load_return_timing_model(),
                )
                predictions.loc[
                    readmitted_mask, "Khoảng thời gian tái nhập viện"
                ] = timing["Khoảng thời gian tái nhập viện"]
                predictions.loc[
                    readmitted_mask,
                    "Xác suất tái nhập viện trong 30 ngày",
                ] = timing["Xác suất tái nhập viện trong 30 ngày"]
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
        except (FileNotFoundError, ValueError, KeyError) as error:
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
    columns = st.columns(3)
    columns[0].metric("Đa số dự đoán tái nhập viện", f"{positive:,}")
    columns[1].metric("Đa số dự đoán không tái nhập viện", f"{negative:,}")
    st.caption(best_model_summary(metrics))
    st.dataframe(results.head(100), width="stretch", hide_index=True)
    st.download_button(
        "Tải kết quả CSV",
        results.to_csv(index=False).encode("utf-8-sig"),
        file_name="ket_qua_du_doan_readmission.csv",
        mime="text/csv",
    )


def render_global_shap(models: dict) -> None:
    st.subheader("SHAP toàn cục (LightGBM)")
    st.write(
        "Mức độ ảnh hưởng tổng thể của từng đặc trưng được đo bằng trung "
        "bình trị tuyệt đối SHAP trên một mẫu ngẫu nhiên đại diện từ dữ liệu "
        "của bài. Giá trị lớn biểu thị ảnh hưởng mạnh hơn; biểu đồ không thể "
        "hiện hướng tác động cho từng giá trị cụ thể."
    )
    st.caption(
        f"Mẫu phân tích gồm tối đa {GLOBAL_SHAP_SAMPLE_SIZE:,} bản ghi, "
        "lấy theo phân bố lớp gốc; nền SHAP gồm 100 bản ghi cân bằng hai lớp. "
        "Các cột one-hot được gộp về đặc trưng ban đầu."
    )

    if st.button("Tính SHAP toàn cục", key="calculate_global_shap"):
        try:
            (
                importance,
                beeswarm,
                inpatient_dependence,
                inpatient_summary,
            ) = explain_global_beeswarm(
                models["LightGBM"], load_shap_global_sample(),
                load_shap_background()
            )
            st.session_state["global_shap_importance"] = importance
            st.session_state["global_shap_beeswarm"] = beeswarm
            st.session_state["inpatient_dependence"] = inpatient_dependence
            st.session_state["inpatient_dependence_summary"] = (
                inpatient_summary
            )
        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(f"Không thể tạo SHAP toàn cục: {error}")
            return

    importance = st.session_state.get("global_shap_importance")
    beeswarm = st.session_state.get("global_shap_beeswarm")
    inpatient_dependence = st.session_state.get("inpatient_dependence")
    inpatient_summary = st.session_state.get(
        "inpatient_dependence_summary"
    )
    if (
        importance is None
        or beeswarm is None
        or inpatient_dependence is None
        or inpatient_summary is None
    ):
        st.info("Nhấn nút trên để tính và xem mức độ ảnh hưởng toàn cục.")
        return

    st.markdown("#### Dependence plot: `number_inpatient`")
    st.caption(
        "Mỗi chấm là một bản ghi trong mẫu 1.000 dòng; trục dọc là đóng góp "
        "SHAP lên xác suất dự đoán (không phải xác suất cuối cùng của bệnh "
        "nhân). Đường đỏ là SHAP trung bình tại từng số lượt: trên 0 làm "
        "tăng, dưới 0 làm giảm xác suất do riêng đặc trưng này."
    )
    st.image(inpatient_dependence, width="stretch")
    inpatient_table = (
        inpatient_summary.rename_axis("Số lượt nhập viện")
        .reset_index()
        .rename(
            columns={
                "SHAP trung bình": "SHAP trung bình (điểm %)"
            }
        )
    )
    inpatient_table["SHAP trung bình (điểm %)"] *= 100
    st.dataframe(
        inpatient_table.style.format(
            {"SHAP trung bình (điểm %)": "{:+.2f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Beeswarm: chiều tác động của giá trị đặc trưng")
    st.caption(
        "Mỗi chấm là một bản ghi; vị trí bên phải/trái thể hiện SHAP làm "
        "tăng/giảm xác suất tái nhập viện. Biểu đồ hiển thị 20 biến đã mã "
        "hóa quan trọng nhất; các biến còn lại được gộp ở hàng cuối. "
        "Đỏ biểu thị giá trị cao, xanh là "
        "thấp. Với biến phân loại one-hot, cao/thấp tương ứng với có/không "
        "thuộc nhóm giá trị ghi trong tên đặc trưng."
    )
    st.image(beeswarm, width="stretch")

    st.markdown("#### Mức độ ảnh hưởng tổng thể")
    chart_data = (
        importance.head(15)
        .mul(100)
        .sort_values()
        .rename_axis("Đặc trưng")
        .reset_index(name="Mean |SHAP| (điểm %)")
    )
    st.bar_chart(
        chart_data,
        x="Mean |SHAP| (điểm %)",
        y="Đặc trưng",
        x_label="Mean |SHAP| (điểm % xác suất)",
        y_label="Đặc trưng",
        horizontal=True,
        sort=False,
    )
    st.dataframe(
        importance.head(15).mul(100)
        .rename("Mean |SHAP| (điểm %)")
        .rename_axis("Đặc trưng")
        .reset_index()
        .style.format({"Mean |SHAP| (điểm %)": "{:.3f}"}),
        width="stretch",
        hide_index=True,
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
individual_tab, batch_tab, global_shap_tab = st.tabs(
    ["Một bệnh nhân", "Tải CSV hàng loạt", "SHAP toàn cục"]
)
with individual_tab:
    render_single_prediction(models, metrics)
with batch_tab:
    render_batch_prediction(models, metrics)
with global_shap_tab:
    render_global_shap(models)
