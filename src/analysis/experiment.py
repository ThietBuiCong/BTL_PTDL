from dataset import fetch_dataset
from train import (
    grid_search_logistic_regression,
    grid_search_random_forest,
    grid_search_xgboost,
    test_model,
    test_random_forest,
    test_xgboost,
)
from clean_dataset import build_preprocessor, clean_features
from train import (
    grid_search_random_forest,
    grid_search_xgboost,
    test_random_forest,
    test_xgboost,
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import os

def luu_du_lieu_scaler(X_scaled, y, preprocessor):
    """
    Lưu dữ liệu sau khi chuẩn hóa thành file CSV.

    Args:
        X_scaled: Features sau khi chuẩn hóa.
        y: Target ban đầu.
        preprocessor: Bộ tiền xử lý đã được fit.
    """

    # Lấy tên các features sau tiền xử lý
    feature_names = preprocessor.get_feature_names_out()

    # Chuyển dữ liệu sau scaler thành DataFrame
    X_df = pd.DataFrame(
        X_scaled,
        columns=feature_names
    )

    y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    X_df["target"] = y_series.reset_index(drop=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "../../data/processed")

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "scaled_data.csv")

    X_df.to_csv(output_path, index=False)

    print("\n========== LƯU DỮ LIỆU ==========")
    print(f"Đã lưu dữ liệu tại: {output_path}")
    print(f"Kích thước dữ liệu: {X_df.shape}")
    return X_df

def thong_ke(X, y):
    """
    Hàm thống kê số lượng dữ liệu
    """

    # Lấy cột Target đầu tiên
    

    # Thống kê tổng quan
    total_samples = len(y)
    total_features = X.shape[1]

    # Đếm số lượng từng nhãn
    label_counts = y.value_counts().sort_index()

    print("========== THỐNG KÊ DATASET ==========")
    print(f"Tổng số mẫu dữ liệu: {total_samples}")
    print(f"Tổng số features: {total_features}")

    print("\n========== THỐNG KÊ TARGET ==========")

    for label, count in label_counts.items():
        percentage = count / total_samples * 100

        print(f"Target {label}: {count} mẫu ({percentage:.2f}%)")

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=label_counts.index.astype(str),
        y=label_counts.values
    )

    plt.title("Phân bố số lượng mẫu theo Target")
    plt.xlabel("Target")
    plt.ylabel("Số lượng mẫu")

    for i, value in enumerate(label_counts.values):
        plt.text(i, value, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def main():
    """
    Hàm main để chạy chương trình
    """

    # Đọc dataset
    X, y = fetch_dataset()
    
    
    
    y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    # Thống kê dữ liệu
    # thong_ke(X, y)

    # Làm sạch giá trị đầu vào trước khi chia tập; fit tiền xử lý trên tập train.
    X = clean_features(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    print("\n========== LOGISTIC REGRESSION ==========")
    grid_search_logistic_regression(preprocessor, X_train, y_train, cv=3)
    model = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "logistic_regression_model.pkl",
    )
    test_model(model, X_test, y_test)

    print("\n========== RANDOM FOREST ==========")
    grid_search_random_forest(preprocessor, X_train, y_train, cv=3)
    test_random_forest(X_test, y_test)

    print("\n========== XGBOOST ==========")
    grid_search_xgboost(preprocessor, X_train, y_train, cv=3)
    test_xgboost(X_test, y_test)


if __name__ == "__main__":
    main()