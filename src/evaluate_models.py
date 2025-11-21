import os
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import tensorflow as tf 
from typing import Tuple

DATA_DIR = "data/processed"
MODEL_DIR = "models"
FIG_DIR = "outputs/figures"

NORMAL_CLASS = 0 


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    X_ae_train = np.load(os.path.join(DATA_DIR, "X_ae_train.npy"))
    return X_train, X_test, y_train, y_test, X_ae_train


def load_models():
    rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    ae_path = os.path.join(MODEL_DIR, "ae_model.h5")

    print(f"[LOAD] RandomForest: {rf_path}")
    rf = joblib.load(rf_path)

    print(f"[LOAD] XGBoost: {xgb_path}")
    xgb = joblib.load(xgb_path)

    ae = None
    if os.path.exists(ae_path):
        print(f"[LOAD] Autoencoder: {ae_path}")
        ae = tf.keras.models.load_model(ae_path)
    else:
        print(f"[WARN] Autoencoder model not found at {ae_path}. Hybrid AE-based detection will be skipped.")

    return rf, xgb, ae


def evaluate_multi_class(model, X_test, y_test, name: str):
    print(f"\n========== {name} – Multi-class Evaluation ==========")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    try:
        # Multi-class ROC-AUC if probabilities available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            roc_auc = roc_auc_score(
                y_test,
                y_prob,
                multi_class="ovr"
            )
            print(f"{name} ROC-AUC (OvR): {roc_auc:.4f}")
    except Exception as e:
        print(f"[WARN] Could not compute {name} ROC-AUC: {e}")


def evaluate_binary(y_true_bin, y_pred_bin, name: str):
    print(f"\n========== {name} – Binary Attack Detection ==========")
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    return acc, prec, rec, f1, cm


def main():
    print("\nSTART -----------Evaluating Models-----------")

    X_train, X_test, y_train, y_test, X_ae_train = load_data()
    rf, xgb, ae = load_models()

    os.makedirs(FIG_DIR, exist_ok=True)

    #evaluation for RF & XGB (known attacks)
    evaluate_multi_class(rf, X_test, y_test, name="RandomForest")
    evaluate_multi_class(xgb, X_test, y_test, name="XGBoost")

    y_true_attack = (y_test != NORMAL_CLASS).astype(int)

    y_pred_xgb = xgb.predict(X_test)
    y_pred_xgb_attack = (y_pred_xgb != NORMAL_CLASS).astype(int)
    evaluate_binary(y_true_attack, y_pred_xgb_attack, name="XGBoost (binary normal vs attack)")

    if ae is None:
        print("\nAutoencoder not available. Skipping autoencoder and hybrid evaluation.")
        print("END -----------Hybrid Evaluation-----------\n")
        return

    print("\nAutoencoder reconstruction error and training ......")
    recon_train = ae.predict(X_ae_train, verbose=0)
    train_errors = np.mean((X_ae_train - recon_train) ** 2, axis=1)
    threshold = train_errors.mean() + 3 * train_errors.std()
    print(f"Autoencoder Anomaly threshold (mean + 3*std): {threshold:.6f}")

    print("Autoencoder Computing reconstruction error on test set...")
    recon_test = ae.predict(X_test, verbose=0)
    test_errors = np.mean((X_test - recon_test) ** 2, axis=1)

    ae_pred_attack = (test_errors > threshold).astype(int)  # 1 = anomaly/attack
    evaluate_binary(y_true_attack, ae_pred_attack, name="Autoencoder (binary anomaly detection)")

    print("\n[Hybrid XGBoost classifier + Autoencoder anomaly detector...")
    hybrid_attack_pred = y_pred_xgb_attack.copy()


    for i in range(len(hybrid_attack_pred)):
        if y_pred_xgb_attack[i] == 0 and ae_pred_attack[i] == 1:
            hybrid_attack_pred[i] = 1

    evaluate_binary(y_true_attack, hybrid_attack_pred, name="Hybrid (XGBoost + AE)")

    print("\nSTART -----------Evaluating Models-----------")


if __name__ == "__main__":
    main()
