import os
import time
import random
from datetime import datetime

import numpy as np
import joblib
import tensorflow as tf

DATA_DIR = "data/processed"
MODEL_DIR = "models"

NORMAL_CLASS = 0       # encoded value for BENIGN/normal
STREAM_DELAY_SEC = 0.5 # delay between flows (adjust for demo)


def load_data():
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    X_ae_train = np.load(os.path.join(DATA_DIR, "X_ae_train.npy"))
    return X_test, y_test, X_ae_train


def load_models_and_encoder():
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    ae_path = os.path.join(MODEL_DIR, "ae_model.h5")
    if not os.path.exists(ae_path):
        raise FileNotFoundError("Autoencoder model not found at models/ae_model.h5")

    ae = tf.keras.models.load_model(ae_path)

    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    return xgb, ae, le


def compute_ae_threshold(ae, X_ae_train):
    recon = ae.predict(X_ae_train, verbose=0)
    train_errors = np.mean((X_ae_train - recon) ** 2, axis=1)
    threshold = train_errors.mean() + 3 * train_errors.std()
    print(f"[AE] Computed anomaly threshold: {threshold:.6f}")
    return threshold


# ---------- NEW: helper to generate fake but realistic IP/port info ----------
def generate_ip_port(idx: int, label_str: str):
    """
    Deterministically generate src/dst IP and ports from index + label.
    This makes the output look realistic, but stays reproducible.
    """
    # Simple internal IP patterns
    src_ip = f"192.168.{(idx // 256) % 256}.{idx % 256}"
    dst_ip = f"10.0.{(idx // 128) % 256}.{(idx * 7) % 256}"

    label_lower = label_str.lower()

    # Default ports
    src_port = 10000 + (idx % 5000)  # ephemeral client port
    dst_port = 4444                  # generic service port

    # Map some common attack / service patterns to typical ports
    if "ssh" in label_lower:
        dst_port = 22
    elif "ftp" in label_lower:
        dst_port = 21
    elif "telnet" in label_lower:
        dst_port = 23
    elif "smtp" in label_lower:
        dst_port = 25
    elif "dns" in label_lower:
        dst_port = 53
    elif "http" in label_lower or "web" in label_lower:
        dst_port = 80
    elif "https" in label_lower:
        dst_port = 443
    elif "ddos" in label_lower or "dos" in label_lower:
        dst_port = 80
    elif "portscan" in label_lower or "scan" in label_lower:
        dst_port = 80
    elif "sql" in label_lower:
        dst_port = 1433

    return src_ip, src_port, dst_ip, dst_port


def simulate_stream():
    print("\nSTART -----------Real-time Stream Simulation-----------")

    X_test, y_test, X_ae_train = load_data()
    xgb, ae, le = load_models_and_encoder()
    threshold = compute_ae_threshold(ae, X_ae_train)

    # Prepare binary ground truth: 0 = normal, 1 = attack
    y_true_attack = (y_test != NORMAL_CLASS).astype(int)

    # We'll stream a random subset of test flows
    n_samples = min(50, len(X_test))   # limit for demo
    indices = list(range(len(X_test)))
    random.shuffle(indices)
    indices = indices[:n_samples]

    print(f"[INFO] Streaming {len(indices)} flows from X_test...")

    for idx in indices:
        x = X_test[idx : idx + 1]         # shape (1, n_features)
        y_true_num = y_test[idx]
        y_true_label = le.inverse_transform([y_true_num])[0]

        # XGBoost prediction
        y_pred_num = xgb.predict(x)[0]
        y_pred_label = le.inverse_transform([y_pred_num])[0]

        # AE reconstruction error
        recon = ae.predict(x, verbose=0)
        err = float(np.mean((x - recon) ** 2))
        ae_attack_flag = 1 if err > threshold else 0

        # Hybrid decision: start from XGB attack vs normal
        xgb_attack_flag = 1 if y_pred_num != NORMAL_CLASS else 0
        hybrid_attack_flag = xgb_attack_flag
        if xgb_attack_flag == 0 and ae_attack_flag == 1:
            hybrid_attack_flag = 1  # AE overrides when it sees anomaly

        # Pretty labels for attack/normal
        true_state = "ATTACK" if y_true_attack[idx] == 1 else "NORMAL"
        xgb_state = "ATTACK" if xgb_attack_flag == 1 else "NORMAL"
        ae_state = "ANOMALY" if ae_attack_flag == 1 else "NORMAL"
        hybrid_state = "ATTACK" if hybrid_attack_flag == 1 else "NORMAL"

        # NEW: generate IP/port info based on index and true label
        src_ip, src_port, dst_ip, dst_port = generate_ip_port(idx, y_true_label)

        # Timestamp for log
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Console log line with IP/port info
        log_line = (
            f"[{ts}] FLOW_ID={idx:06d} "
            f"src={src_ip}:{src_port} -> dst={dst_ip}:{dst_port} | "
            f"True={true_state}({y_true_label}) | "
            f"XGB={xgb_state}({y_pred_label}) | "
            f"AE_err={err:.6f} -> {ae_state} | "
            f"HYBRID={hybrid_state}"
        )

        print(log_line)
        time.sleep(STREAM_DELAY_SEC)

    print("\nEND -----------Real-time Stream Simulation-----------\n")


if __name__ == "__main__":
    simulate_stream()
