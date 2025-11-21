import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

DATA_DIR = "data/processed"
MODEL_DIR = "models"

def main():
    print("\nSTART -----------Hybrid Model-----------")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")
    X_ae_train = np.load(f"{DATA_DIR}/X_ae_train.npy")

    rf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    xgb = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    ae = tf.keras.models.load_model(f"{MODEL_DIR}/ae_model.h5")

    
    y_pred_ml = xgb.predict(X_test)

    # Autoencoder reconstruction error
    recon = ae.predict(X_test, verbose=0)
    errors = np.mean((X_test - recon)**2, axis=1)

    recon_train = ae.predict(X_ae_train, verbose=0)
    train_errors = np.mean((X_ae_train - recon_train)**2, axis=1)
    threshold = train_errors.mean() + 3 * train_errors.std()
    print(f"Threshold for anomaly: {threshold:.6f}")


    NORMAL_CLASS = 0
    hybrid_pred = y_pred_ml.copy()

    for i in range(len(y_pred_ml)):
        if y_pred_ml[i] == NORMAL_CLASS and errors[i] > threshold:
            
            hybrid_pred[i] = -1 

    print("\nEND -----------Autoencoder Model-----------")
    print("Evaluation report:")
    print(classification_report(y_test, y_pred_ml))  

if __name__ == "__main__":
    main()
