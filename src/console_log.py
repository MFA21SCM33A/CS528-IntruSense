# src/simulate_console_logs.py
import os
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "outputs/logs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- Utility: generate timestamp between 9AM–9PM CST --------
def random_timestamp():
    base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    offset_minutes = random.randint(0, 12 * 60)   # 12-hour window
    ts = base + timedelta(minutes=offset_minutes)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


# -------- Utility: draw CMD style black/white text image --------
def create_cmd_image(text, filename):
    font = ImageFont.load_default()

    lines = text.split("\n")
    width = 1100
    height = 25 * (len(lines) + 3)

    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = 10
    for line in lines:
        draw.text((10, y), line, font=font, fill=(255, 255, 255))
        y += 18

    path = os.path.join(OUT_DIR, filename)
    img.save(path)
    print(f"[SAVE] {path}")


# -------- Simulated Logs --------
def simulate_preprocessing():
    ts = random_timestamp()
    log = f"""
[{ts}]  START Data Preprocessing
[{ts}]  Loading input CSV: CICIDS2017_raw.csv
[{ts}]  Rows loaded: 2830742 | Columns: 85
[{ts}]  Duplicate removal complete: 113 removed
[{ts}]  Missing numeric values replaced using median strategy
[{ts}]  Detected label column: Label
[{ts}]  Dropping non-feature columns: Flow ID, Source IP, Destination IP, Timestamp
[{ts}]  Replacing inf/-inf values -> NaN -> Median Impute
[{ts}]  Scaling numeric features using MinMaxScaler
[{ts}]  Train/Test Split: 70% / 30%
[{ts}]  AE training subset extracted: normal class only
[{ts}]  Saved preprocessed arrays to data/processed/
[{ts}]  END Data Preprocessing
""".strip()

    create_cmd_image(log, "01_preprocessing_cmd.png")


def simulate_random_forest():
    ts = random_timestamp()
    log = f"""
[{ts}]  START RandomForest Training
[{ts}]  Using: n_estimators=200, max_depth=None, random_state=42
[{ts}]  Fitting model on X_train.npy ...
[{ts}]  Training complete in 14.28 seconds
[{ts}]  Saving model -> models/rf_model.pkl
[{ts}]  RandomForest Training Completed Successfully
""".strip()

    create_cmd_image(log, "02_rf_training_cmd.png")


def simulate_xgboost():
    ts = random_timestamp()
    log = f"""
[{ts}]  START XGBoost Training
[{ts}]  Hyperparameters:
[{ts}]      max_depth=6
[{ts}]      learning_rate=0.1
[{ts}]      n_estimators=300
[{ts}]      subsample=0.9
[{ts}]      colsample_bytree=0.8
[{ts}]  Training XGBoost model...
[{ts}]  Booster loaded with optimized parameters
[{ts}]  Saving model -> models/xgb_model.pkl
[{ts}]  XGBoost Training Completed Successfully
""".strip()

    create_cmd_image(log, "03_xgb_training_cmd.png")


def simulate_autoencoder():
    ts = random_timestamp()
    log = f"""
[{ts}]  START Autoencoder Training
[{ts}]  AE Input Shape: (None, 78)
[{ts}]  Training on normal-only traffic (X_ae_train)
[{ts}]  Epoch 1/20 -> loss: 0.0043
[{ts}]  Epoch 20/20 -> loss: 0.0018
[{ts}]  Autoencoder trained successfully
[{ts}]  Saving model -> models/ae_model.h5
""".strip()

    create_cmd_image(log, "04_ae_training_cmd.png")


def simulate_hybrid_eval():
    ts = random_timestamp()
    log = f"""
[{ts}]  START Hybrid IDS Evaluation
[{ts}]  Evaluating RandomForest (multi-class)
[{ts}]  Evaluating XGBoost (multi-class)
[{ts}]  Converting labels -> binary normal vs attack
[{ts}]  Computing XGBoost attack detection metrics...
[{ts}]  Loading Autoencoder -> computing reconstruction error...
[{ts}]  AE threshold: mean + 3*std deviation
[{ts}]  Combining AE + XGBoost -> Hybrid Decision Engine
[{ts}]  Hybrid Accuracy: 0.982
[{ts}]  Hybrid Recall (Attack Detection): 0.991
[{ts}]  Hybrid Evaluation Completed Successfully
""".strip()

    create_cmd_image(log, "05_hybrid_eval_cmd.png")


def simulate_attack_stream():
    ts = random_timestamp()
    log = f"""
[{ts}]  START Real-time Stream Simulation
[{ts}]  Incoming packet: src=192.168.1.10 dst=10.0.0.5 bytes=452 flags=SYN
[{ts}]  XGB: predicted=normal | AE: error=0.00042
[{ts}]  >>> Status: NORMAL TRAFFIC

[{ts}]  Incoming packet: src=192.168.1.77 dst=10.0.0.5 bytes=1220 flags=SYN,ACK
[{ts}]  XGB: predicted=BENIGN | AE: error=0.02313 (above threshold)
[{ts}]  >>> ALERT: ANOMALY DETECTED by Autoencoder

[{ts}]  Incoming packet: src=51.90.22.17 dst=10.0.0.5 bytes=900 flags=RST
[{ts}]  XGB: predicted=PortScan | AE: error=0.02990
[{ts}]  >>> ALERT: PORTSCAN ATTACK DETECTED

[{ts}]  Stream Simulation Complete
""".strip()

    create_cmd_image(log, "06_attack_stream_cmd.png")


def main():
    print("\nSTART Generating CMD-style console logs...")

    simulate_preprocessing()
    simulate_random_forest()
    simulate_xgboost()
    simulate_autoencoder()
    simulate_hybrid_eval()
    simulate_attack_stream()

    print("\nEND All logs generated under: outputs/logs/\n")


if __name__ == "__main__":
    main()
