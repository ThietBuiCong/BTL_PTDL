from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "hospital_readmission.csv"
OUTPUT_DIR = SCRIPT_DIR.parent / "asset_data"
TARGET_COLUMN = "readmitted_30days"


def _read_raw_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Raw dataset was not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def _save_correlation_heatmap(raw_data):
    numeric_data = raw_data.apply(pd.to_numeric, errors="coerce")
    correlation = numeric_data.corr()
    figure, axis = plt.subplots(figsize=(11, 9))
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
    axis.set_title("Raw Data Correlation Heatmap (before cleaning)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "raw_correlation_heatmap.png", dpi=160)
    plt.close(figure)


def _save_raw_feature_plots(raw_data):
    feature_columns = [
        column for column in raw_data.columns if column != TARGET_COLUMN
    ]
    columns = 3
    rows = (len(feature_columns) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, rows * 4),
        squeeze=False,
    )

    for axis, column in zip(axes.flat, feature_columns):
        values = pd.to_numeric(raw_data[column], errors="coerce").dropna()
        if values.empty:
            axis.set_title(f"{column} (no numeric values)")
            axis.set_axis_off()
            continue
        sns.histplot(values, bins=30, kde=values.nunique() > 10, ax=axis)
        axis.set_title(f"{column} distribution")
        axis.set_xlabel(column)
        axis.set_ylabel("Count")

    for axis in axes.flat[len(feature_columns):]:
        axis.set_visible(False)

    figure.suptitle("Raw Feature Distributions (before cleaning)", y=1.01)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "raw_feature_distributions.png", dpi=160)
    plt.close(figure)


def _save_category_bar_chart(raw_data):
    category_columns = [
        column
        for column in ("has_diabetes", "discharge_type", TARGET_COLUMN)
        if column in raw_data.columns
    ]
    figure, axes = plt.subplots(
        1,
        len(category_columns),
        figsize=(6 * len(category_columns), 5),
        squeeze=False,
    )

    for axis, column in zip(axes.flat, category_columns):
        labels = raw_data[column].map(
            lambda value: "Missing" if pd.isna(value) else str(value)
        )
        counts = labels.value_counts().sort_index()
        sns.barplot(x=counts.index, y=counts.values, ax=axis, color="steelblue")
        axis.set_title(f"{column} counts (raw)")
        axis.set_xlabel(column)
        axis.set_ylabel("Count")
        axis.tick_params(axis="x", rotation=30)
        for position, count in enumerate(counts.values):
            axis.text(position, count, f"{count:,}", ha="center", va="bottom")

    figure.suptitle("Raw Category Counts (including missing values)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "raw_category_bar.png", dpi=160)
    plt.close(figure)


def _save_target_pie_chart(raw_data):
    target_counts = raw_data[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    labels = [
        "Missing" if pd.isna(value) else f"Readmitted = {int(value)}"
        for value in target_counts.index
    ]
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.pie(
        target_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    axis.set_title("Raw Readmission Target Proportions")
    axis.axis("equal")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "raw_readmission_pie.png", dpi=160)
    plt.close(figure)


def create_raw_data_visualizations():
    """Create visualizations directly from the uncleaned source CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_data = _read_raw_data()

    _save_correlation_heatmap(raw_data)
    _save_raw_feature_plots(raw_data)
    _save_category_bar_chart(raw_data)
    _save_target_pie_chart(raw_data)

    print(f"Raw-data visualizations saved in: {OUTPUT_DIR}")
    print(
        f"Source rows: {len(raw_data):,}; "
        f"missing cells: {int(raw_data.isna().sum().sum()):,}"
    )


if __name__ == "__main__":
    create_raw_data_visualizations()
