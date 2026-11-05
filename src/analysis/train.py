from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import os
import joblib
from xgboost import XGBClassifier


def _model_path(filename):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, filename)


def _train_and_save(preprocessor, classifier, filename, model_name, X_train, y_train):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    model.fit(X_train, y_train)

    path = _model_path(filename)
    joblib.dump(model, path)
    print(f"==Success-{model_name}==")
    print(f"Saved model: {path}")
    return model


def train_model(preprocessor, X_train, y_train):
    """
        Huấn luyện mô hình Logistic
    """
    pipeline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
        # ('classifier', XGBClassifier(
        # #n_estimators=400,
        # #learning_rate=0.05,
        # #max_depth=6,
        # subsample=0.8,
        # #colsample_bytree=0.8,
        # gamma=1,
        # eval_metric="logloss",
        # random_state=42))
    ])
    
    #Huấn luyện mô hình
    pipeline_model.fit(X_train, y_train)
    
    print("==Success-Logistic Regression==")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join('..', current_dir, 'models', 'logistic_regression_model.pkl')
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(pipeline_model, model_filename)

def test_model(model, X_test, y_test):
    """
    Kiểm thử model

    Args:
        model (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
    """
    model = joblib.load(model)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['No Readmission', 'Readmission']))


def train_random_forest(preprocessor, X_train, y_train):
    """Huấn luyện và lưu mô hình Random Forest."""
    return _train_and_save(
        preprocessor,
        RandomForestClassifier(random_state=42, n_jobs=-1),
        "random_forest_model.pkl",
        "Random Forest",
        X_train,
        y_train,
    )


def train_xgboost(preprocessor, X_train, y_train):
    """Huấn luyện và lưu mô hình XGBoost."""
    return _train_and_save(
        preprocessor,
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost_model.pkl",
        "XGBoost",
        X_train,
        y_train,
    )


def _grid_search(preprocessor, classifier, param_grid, filename, model_name,
                 X_train, y_train, cv):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    path = _model_path(filename)
    joblib.dump(search.best_estimator_, path)
    print(f"==Best-{model_name}==")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Saved best model: {path}")
    return search


def grid_search_random_forest(preprocessor, X_train, y_train, cv=3):
    """Tìm tham số tốt nhất cho Random Forest và lưu estimator tốt nhất."""
    return _grid_search(
        preprocessor,
        RandomForestClassifier(random_state=42, n_jobs=1),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_split": [2, 5],
        },
        "random_forest_model.pkl",
        "Random Forest",
        X_train,
        y_train,
        cv,
    )


def grid_search_logistic_regression(preprocessor, X_train, y_train, cv=3):
    """Tìm tham số tốt nhất cho Logistic Regression và lưu estimator tốt nhất."""
    return _grid_search(
        preprocessor,
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__l1_ratio": [0, 1],
        },
        "logistic_regression_model.pkl",
        "Logistic Regression",
        X_train,
        y_train,
        cv,
    )


def grid_search_xgboost(preprocessor, X_train, y_train, cv=3):
    """Tìm tham số tốt nhất cho XGBoost và lưu estimator tốt nhất."""
    return _grid_search(
        preprocessor,
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        },
        "xgboost_model.pkl",
        "XGBoost",
        X_train,
        y_train,
        cv,
    )


def test_random_forest(X_test, y_test):
    """Đánh giá Random Forest đã huấn luyện hoặc GridSearchCV lưu."""
    test_model(_model_path("random_forest_model.pkl"), X_test, y_test)


def test_xgboost(X_test, y_test):
    """Đánh giá XGBoost đã huấn luyện hoặc GridSearchCV lưu."""
    test_model(_model_path("xgboost_model.pkl"), X_test, y_test)
