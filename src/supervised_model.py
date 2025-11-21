import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import os

DATA_DIR = "data/processed"
MODEL_DIR = "models"

def load_data():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        criterion="gini",
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print("Random Forest valuation on test set:")
    print(classification_report(y_test, rf.predict(X_test)))
    return rf

def train_xgboost(X_train, y_train, X_test, y_test):
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    xgb.fit(X_train, y_train)
    print("XGBoost Evaluation on test set:")
    print(classification_report(y_test, xgb.predict(X_test)))
    return xgb

def main():
    print("\nSTART -----------Generating Supervised Model-----------")
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    rf = train_random_forest(X_train, y_train, X_test, y_test)
    xgb = train_xgboost(X_train, y_train, X_test, y_test)

    joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")
    joblib.dump(xgb, f"{MODEL_DIR}/xgb_model.pkl")
    print("\nEND -----------Generating Supervised Model-----------")

if __name__ == "__main__":
    main()
