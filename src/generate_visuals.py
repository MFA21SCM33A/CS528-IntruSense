# src/generate_visuals.py
import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)
import tensorflow as tf

DATA_DIR = "data/processed"
MODEL_DIR = "models"
FIG_DIR = "outputs/figures"

NORMAL_CLASS = 0  # same as in preprocess/evaluate


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    X_ae_train = None
    x_ae_path = os.path.join(DATA_DIR, "X_ae_train.npy")
    if os.path.exists(x_ae_path):
        X_ae_train = np.load(x_ae_path)
    return X_train, X_test, y_train, y_test, X_ae_train


def load_models():
    rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))

    ae_path = os.path.join(MODEL_DIR, "ae_model.h5")
    ae = None
    if os.path.exists(ae_path):
        ae = tf.keras.models.load_model(ae_path)
    return rf, xgb, ae


def plot_confusion_matrix_xgb(xgb, X_test, y_test):
    y_pred = xgb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("XGBoost – Confusion Matrix (Multi-class)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "xgb_confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_binary_roc_xgb(xgb, X_test, y_test):
    # Binary: normal vs attack
    y_true_attack = (y_test != NORMAL_CLASS).astype(int)

    if not hasattr(xgb, "predict_proba"):
        print("[WARN] XGBoost has no predict_proba. Skipping ROC curve.")
        return

    y_proba = xgb.predict_proba(X_test)
    # Probability of "attack" = 1 - P(normal) if class 0 = normal
    # We assume label encoding: normal = NORMAL_CLASS
    if NORMAL_CLASS in xgb.classes_:
        normal_index = list(xgb.classes_).index(NORMAL_CLASS)
        p_normal = y_proba[:, normal_index]
        p_attack = 1.0 - p_normal
    else:
        # Fallback: just use max prob (not perfect, but safe)
        p_attack = 1.0 - np.max(y_proba, axis=1)

    fpr, tpr, _ = roc_curve(y_true_attack, p_attack)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Normal vs Attack (XGBoost)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "xgb_roc_curve_binary.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_feature_importance(model, feature_names, name, top_n=20, filename="feature_importance.png"):
    if not hasattr(model, "feature_importances_"):
        print(f"[WARN] Model {name} has no feature_importances_. Skipping feature importance plot.")
        return

    importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]

    idx = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[idx]
    top_features = [feature_names[i] for i in idx]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, orient="h")
    plt.title(f"{name} – Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_ae_reconstruction_hist(ae, X_ae_train, X_test):
    if ae is None or X_ae_train is None:
        print("[INFO] AE or X_ae_train not available. Skipping AE reconstruction plot.")
        return

    recon_train = ae.predict(X_ae_train, verbose=0)
    train_errors = np.mean((X_ae_train - recon_train) ** 2, axis=1)

    recon_test = ae.predict(X_test, verbose=0)
    test_errors = np.mean((X_test - recon_test) ** 2, axis=1)

    plt.figure(figsize=(8, 6))
    sns.histplot(train_errors, bins=50, color="blue", label="Train (normal)", stat="density", kde=True)
    sns.histplot(test_errors, bins=50, color="red", label="Test (mixed)", stat="density", kde=True, alpha=0.5)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("Autoencoder Reconstruction Error – Train vs Test")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "ae_reconstruction_error_hist.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] {out_path}")


def main():
    print("\nSTART -----------Generate Visuals-----------")

    os.makedirs(FIG_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, X_ae_train = load_data()
    rf, xgb, ae = load_models()

    feature_names = None

    # Confusion matrix (multi-class) for XGBoost
    plot_confusion_matrix_xgb(xgb, X_test, y_test)

    # ROC curve (binary attack detection)
    plot_binary_roc_xgb(xgb, X_test, y_test)

    # Feature importance for RF & XGB
    plot_feature_importance(
        rf,
        feature_names,
        name="RandomForest",
        top_n=20,
        filename="rf_feature_importance.png"
    )

    plot_feature_importance(
        xgb,
        feature_names,
        name="XGBoost",
        top_n=20,
        filename="xgb_feature_importance.png"
    )

    # Autoencoder reconstruction error histogram
    plot_ae_reconstruction_hist(ae, X_ae_train, X_test)

    print("\nEND -----------Generate Visuals-----------\n")


if __name__ == "__main__":
    main()
