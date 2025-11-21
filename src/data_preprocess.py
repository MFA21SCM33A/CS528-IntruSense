import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import joblib


INPUT_PATH = "data/raw/CICIDS2017_raw.csv"
OUT_DIR = "data/processed"
MODEL_DIR = "models" 

def main():
    print("\nSTART -----------Data Preprocessing-----------")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loading input data: rows[{df.shape[0]}], columns[{df.shape[1]}]")

    # Remove duplicate records
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Removed duplicate Records: [ {before - df.shape[0]} ]")

    # Handle missing numeric data (initial pass)
    df = df.fillna(df.median(numeric_only=True))
    print("Missing numeric values replaced with median")

    #Detect label column (CICIDS2017 vs NSL-KDD)
    label_col = None
    for cand in [" Label", "Label", "label"]:
        if cand in df.columns:
            label_col = cand
            break

    if label_col is None:
        raise KeyError(
            f"No label column found. "
            f"Looked for ' Label', 'Label', 'label'. "
            f"Actual columns: {list(df.columns)}"
        )

    print(f"Using label column [{label_col}]")

    # Drop known non-feature columns from dataset
    drop_cols = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "SourceFile",        # from CICIDS
        "filename",
        "difficulty_level"   # NSL-KDD extra column
    ]
    for c in drop_cols:
        if c in df.columns and c != label_col:
            df = df.drop(columns=[c])
            print(f"Dropped non-feature column: [{c}]")

    #Encode label column
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(" Saved label encoder -> models/label_encoder.pkl")

    # Separate features and target
    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Clean infinities and very large values before scaling 
    X = X.replace([np.inf, -np.inf], np.nan)

    # Keep only numeric columns for scaling (drops any leftover strings)
    num_cols = X.select_dtypes(include=["number"]).columns
    print(f"Numeric feature columns count: {len(num_cols)}")

    # Fill any remaining NaNs in numeric features with median
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # Scale numeric features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[num_cols])
    print("Numeric features scaled with MinMaxScaler")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )

    # Autoencoder training data = only normal traffic (label 0, e.g. BENIGN)
    NORMAL_CLASS = 0
    X_ae_train = X_train[y_train == NORMAL_CLASS]

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(OUT_DIR, "X_ae_train.npy"), X_ae_train)

    print("\nEND -----------Data Preprocessing-----------\n")

if __name__ == "__main__":
    main()
