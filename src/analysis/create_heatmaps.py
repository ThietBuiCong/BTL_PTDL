from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "asset_data"
FEATURE_NAMES = [
    "age",
    "length_of_stay_days",
    "num_diagnoses",
    "num_medications",
    "prev_admissions",
    "glucose_level",
    "bmi",
    "has_diabetes",
    "discharge_type",
]


def _read_numeric_data(csv_path):
    data = pd.read_csv(csv_path)

    if len(data.columns) == len(FEATURE_NAMES) and all(
        str(column).isdigit() for column in data.columns
    ):
        data.columns = FEATURE_NAMES

    numeric_data = data.apply(pd.to_numeric, errors="coerce")
    numeric_data = numeric_data.dropna(axis=1, how="all")
    if numeric_data.empty:
        raise ValueError(f"No numeric columns found in {csv_path}")

    return numeric_data


def _save_correlation_heatmap(csv_path, data):
    correlation = data.corr()
    figure_size = max(8, len(correlation.columns) * 0.9)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size * 0.8))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=axis,
    )
    axis.set_title(f"Feature Correlation Heatmap - {csv_path.stem}")
    figure.tight_layout()
    output_path = OUTPUT_DIR / f"{csv_path.stem}_heatmap.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_heatmaps():
    """Create one numeric correlation heatmap for each CSV in src/data."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV datasets found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        numeric_data = _read_numeric_data(csv_path)
        output_path = _save_correlation_heatmap(csv_path, numeric_data)
        print(f"Saved heatmap for {csv_path.name}: {output_path}")


if __name__ == "__main__":
    create_heatmaps()
