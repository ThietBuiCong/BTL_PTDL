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

## Dự đoán hàng loạt

Giao diện nhận đầy đủ các thuộc tính đầu vào của `diabetic_data.csv`, ngoại
trừ hai mã định danh `encounter_id`, `patient_nbr` và nhãn `readmitted`.
Có thể tải CSV mẫu từ giao diện. Các cột ID nhập dưới dạng mã số hoặc mô tả
trong `IDS_mapping.csv`; nếu CSV có `readmitted`, cột đó sẽ bị bỏ qua. Kết quả
CSV có dự đoán và xác suất của từng mô hình cùng kết luận tổng hợp theo đa số
phiếu. Trường hợp hai mô hình dự đoán mỗi lớp được ghi là không đồng thuận.

## Thành viên

1. Trần Thị Thanh Thảo
2. Bùi Công Thiệt
