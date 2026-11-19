# BTL_PTDL

Ứng dụng phân tích và dự đoán tái nhập viện của bệnh nhân tiểu đường.

## Cài đặt và chạy

Từ thư mục gốc dự án, cài đặt thư viện:

```powershell
python -m pip install -r requirements.txt
```

Huấn luyện bốn mô hình và khởi chạy giao diện:

```powershell
python src\frontend\train_models.py
streamlit run src\frontend\app.py
```

Mô hình và kết quả đánh giá được lưu trong `code\models`. Script huấn luyện
dùng bộ dữ liệu gốc `src\data\diabetic_data.csv`. Các mã
`admission_type_id`, `discharge_disposition_id` và `admission_source_id` được
chuyển thành mô tả từ `src\data\IDS_mapping.csv` trước khi huấn luyện và dự đoán.
Ngoài bốn mô hình dự đoán tái nhập viện, script lưu riêng
`code\models\return_timing.joblib` để dự đoán thời điểm khi kết luận tổng hợp
có tái nhập viện. Dữ liệu chỉ cung cấp nhãn `<30` và `>30`, nên đầu ra là
khoảng “trong vòng 30 ngày” hoặc “sau hơn 30 ngày”, không phải ngày chính xác.
Mô hình thời điểm không được gọi nếu kết luận là không tái nhập viện hoặc các
mô hình chưa đồng thuận.

## Dự đoán hàng loạt

Giao diện nhận đầy đủ các thuộc tính đầu vào của `diabetic_data.csv`, ngoại
trừ hai mã định danh `encounter_id`, `patient_nbr` và nhãn `readmitted`.
Có thể tải CSV mẫu từ giao diện. Các cột ID nhập dưới dạng mã số hoặc mô tả
trong `IDS_mapping.csv`; nếu CSV có `readmitted`, cột đó sẽ bị bỏ qua. Kết quả
CSV có dự đoán và xác suất của từng mô hình cùng kết luận tổng hợp theo đa số
phiếu. Trường hợp hai mô hình dự đoán mỗi lớp được ghi là không đồng thuận.
Ở tab một bệnh nhân, giao diện còn hiển thị SHAP cục bộ của LightGBM: đóng
góp của từng đặc trưng vào xác suất tái nhập viện, tính bằng điểm phần trăm
so với xác suất nền của 100 bản ghi lấy cân bằng giữa hai lớp, kèm biểu đồ
Waterfall để thấy từng đóng góp đẩy xác suất từ mức nền tới dự đoán.
Tab SHAP toàn cục tính trung bình trị tuyệt đối SHAP trên mẫu tối đa 1.000
bản ghi ngẫu nhiên từ dữ liệu của bài, giữ phân bố lớp gốc để xếp hạng mức độ
ảnh hưởng của các đặc trưng. Biểu đồ Beeswarm cho 20 đặc trưng đã mã hóa
quan trọng nhất biết giá trị cao/thấp tác động theo hướng nào; màu đỏ/xanh
tương ứng giá trị cao/thấp, và với cột one-hot biểu thị có/không thuộc nhóm
giá trị trong tên cột. Tab cũng có dependence plot cho `number_inpatient`,
thể hiện SHAP theo số lượt nhập viện cùng mức đóng góp SHAP trung bình tại
mỗi số lượt.

## Phân tích hai bài toán tái nhập viện

Để chạy quy trình nghiên cứu mà không khởi chạy giao diện, mở
`code/main_uci.ipynb`, chạy cell nạp dữ liệu đầu tiên, sau đó chạy cell cuối
“Quy trình mô hình hóa đã hiệu chỉnh”. Logic huấn luyện và đánh giá nằm trong
`src/research_analysis.py`. Quy trình dùng chung phép chia theo `patient_nbr`,
chọn đặc trưng trên validation và chỉ dùng test holdout cho đánh giá cuối cùng.

Để tính AUC và Recall theo `race`, `gender`, `age` cùng khoảng tin cậy
bootstrap 95%, chạy `python src\frontend\evaluate_subgroups.py`. Phân tích
dùng holdout chia theo bệnh nhân, mô hình LightGBM và ngưỡng dự đoán 0,5;
khoảng tin cậy bootstrap lấy mẫu theo cụm bệnh nhân. Kết quả được lưu ở
`code\subgroup_metrics.csv`.

## Thành viên

1. Trần Thị Thanh Thảo
2. Bùi Công Thiệt
