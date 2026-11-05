import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from clean_dataset import build_preprocessor, clean_features
from dataset import fetch_dataset


RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "asset",
)


def _build_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _get_feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = [
        name.split("__", 1)[-1]
        for name in preprocessor.get_feature_names_out()
    ]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_).mean(axis=0)
    else:
        raise TypeError(
            f"Unsupported classifier for feature importance: "
            f"{type(classifier).__name__}"
        )

    if len(feature_names) != len(importances):
        raise ValueError("Feature names and importance values have different lengths.")
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def _save_confusion_matrix(y_true, y_pred, strategy, model_name):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No Readmission", "Readmission"],
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    display.ax_.set_title(f"Confusion Matrix - {model_name} ({strategy})")
    display.figure_.tight_layout()
    display.figure_.savefig(
        os.path.join(OUTPUT_DIR, f"confusion_matrix_{strategy}_{model_name}.png"),
        dpi=160,
    )
    plt.close(display.figure_)


def _save_feature_importance(model, strategy, model_name):
    importance = _get_feature_importance(model).head(15).sort_values()
    figure_height = max(4, 0.35 * len(importance))
    figure, axis = plt.subplots(figsize=(9, figure_height))
    sns.barplot(x=importance.values, y=importance.index, ax=axis, color="steelblue")
    axis.set(
        title=f"Top 15 Feature Importance - {model_name} ({strategy})",
        xlabel="Importance (absolute coefficient for Logistic Regression)",
        ylabel="Feature",
    )
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, f"feature_importance_{strategy}_{model_name}.png"),
        dpi=160,
    )
    plt.close(figure)


def _evaluation_metadata(model, y_true, y_pred, y_score, strategy, model_name):
    classifier = model.named_steps["classifier"]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    importance = _get_feature_importance(model)
    return {
        "imputation_strategy": strategy,
        "model": model_name,
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "classes": {
                "0": {
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1_score": float(f1_score[0]),
                },
                "1": {
                    "precision": float(precision[1]),
                    "recall": float(recall[1]),
                    "f1_score": float(f1_score[1]),
                },
            },
        },
        "confusion_matrix": {
            "labels": ["No Readmission", "Readmission"],
            "values": matrix.tolist(),
        },
        "classifier_parameters": classifier.get_params(deep=False),
        "top_features": [
            {"feature": feature, "importance": float(value)}
            for feature, value in importance.head(15).items()
        ],
        "artifacts": {
            "confusion_matrix": f"confusion_matrix_{strategy}_{model_name}.png",
            "feature_importance": f"feature_importance_{strategy}_{model_name}.png",
            "roc_curve": f"roc_auc_{strategy}.png",
        },
    }


def _save_roc_curve(y_true, results, strategy):
    figure, axis = plt.subplots(figsize=(8, 6))
    for model_name, y_score in results.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true,
            y_score,
            pos_label=1,
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC = 0.500)")
    axis.set(
        title=f"ROC Curves - {strategy}",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT_DIR, f"roc_auc_{strategy}.png"), dpi=160)
    plt.close(figure)


def generate_evaluation_images():
    """Train and evaluate each model with mean and most-frequent imputation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y = fetch_dataset()
    y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    X = clean_features(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "file": "hospital_readmission.csv",
            "samples": int(len(X)),
            "features": list(X.columns),
            "target": str(y.name),
            "target_labels": {"0": "No Readmission", "1": "Readmission"},
        },
        "evaluation": {
            "test_size": 0.2,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "stratified": True,
            "random_state": RANDOM_STATE,
            "positive_class": 1,
        },
        "versions": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "results": [],
    }

    for strategy in ("mean", "most_frequent"):
        roc_scores = {}
        for model_name, classifier in _build_models().items():
            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor(X_train, strategy)),
                ("classifier", classifier),
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_score = pipeline.predict_proba(X_test)[:, 1]

            _save_confusion_matrix(y_test, y_pred, strategy, model_name)
            _save_feature_importance(pipeline, strategy, model_name)
            roc_scores[model_name] = y_score
            metadata["results"].append(
                _evaluation_metadata(
                    pipeline,
                    y_test,
                    y_pred,
                    y_score,
                    strategy,
                    model_name,
                )
            )

        _save_roc_curve(y_test, roc_scores, strategy)

    metadata_path = os.path.join(OUTPUT_DIR, "evaluation_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    print(f"Evaluation images saved in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Evaluation metadata saved to: {metadata_path}")


if __name__ == "__main__":
    generate_evaluation_images()
