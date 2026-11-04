from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import os
import joblib
from xgboost import XGBClassifier

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
    print(classification_report(y_test, y_pred, target_names=['Tiểu đường', 'No Tiểu đường']))