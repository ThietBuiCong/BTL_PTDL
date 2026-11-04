import pandas as pd
import os


def fetch_dataset():
    """
    Đọc bộ dữ liệu từ file CSV có sẵn.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            X: Các features
            y: Target
    """

    # Xác định đường dẫn file CSV
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(
        CURRENT_DIR,
        "..",
        "data",
        "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    )

    # Đọc dữ liệu
    df = pd.read_csv(DATA_PATH)

    # Tách Features và Target
    X = df.drop(columns=["Diabetes_binary"])
    y = df[["Diabetes_binary"]]

    return X, y