import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from dataset import fetch_dataset


def clean_features(X):
    """Chuẩn hóa các giá trị đầu vào không hợp lệ trước khi chia tập dữ liệu."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    X["age"] = X["age"].astype(str)
    X["age"] = X["age"].str.replace("_err", "", regex=False)
    X["age"] = pd.to_numeric(X["age"], errors='coerce')

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_features:
        X.loc[X[col] < 0, col] = None

    return X


def build_preprocessor(X, imputation_strategy="mean"):
    """Tạo bộ tiền xử lý dựa trên các cột và chiến lược điền khuyết."""
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=imputation_strategy)),
        ("scaler", MinMaxScaler(feature_range=(-1, 1)))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="passthrough"
    )

    return preprocessor


def scaler(X):
    """Tiền xử lý và chuẩn hóa dữ liệu theo bộ chuẩn hóa đã fit trên X."""
    X = clean_features(X)
    preprocessor = build_preprocessor(X)
    X_scaled = preprocessor.fit_transform(X)
    
    return X_scaled, preprocessor

def clean_data(X: pd.DataFrame, y: pd.Series):
    """
        Hàm để tiền xử lý các dữ liệu bị thiếu, khuyết và sai lệch dữ liệu
    """
    pass
    
if __name__ == "__main__":
    X, y = fetch_dataset()
    X_cleaned, preprocessor = scaler(X)
    
    
    