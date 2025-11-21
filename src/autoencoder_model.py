import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

DATA_DIR = "data/processed"
MODEL_DIR = "models"

def build_autoencoder(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(40, activation="relu")(inp)
    x = layers.Dense(20, activation="relu")(x)
    latent = layers.Dense(10, activation="relu")(x)
    x = layers.Dense(20, activation="relu")(latent)
    x = layers.Dense(40, activation="relu")(x)
    out = layers.Dense(input_dim, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    print("\nSTART -----------Autoencoder Model-----------")
    X_ae_train = np.load(f"{DATA_DIR}/X_ae_train.npy")
    input_dim = X_ae_train.shape[1]

    ae = build_autoencoder(input_dim)
    print("Autoencoder for normal traffic ...")
    ae.fit(
        X_ae_train, X_ae_train,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        verbose=1
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    ae.save(f"{MODEL_DIR}/ae_model.h5")
    print("\nEND -----------Autoencoder Model-----------")

if __name__ == "__main__":
    main()
