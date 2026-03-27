import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from preprocess import FEATURE_COLS, LABEL_COL, load_and_preprocess_dataset

TIME_STEPS = 10
TRAIN_RATIO = 0.8

DATASET_PATH = Path("weather.csv")
MODEL_PATH = Path("lstm_model.h5")
PREPROCESSOR_PATH = Path("preprocessor.pkl")
METRICS_PATH = Path("test_metrics.json")


def build_train_test_sequences(df, scaler: MinMaxScaler):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for _, loc_df in df.groupby("Location", sort=False):
        loc_df = loc_df.sort_values("Date").reset_index(drop=True)
        n = len(loc_df)
        if n <= TIME_STEPS + 1:
            continue

        split_idx = max(TIME_STEPS + 1, int(n * TRAIN_RATIO))
        split_idx = min(split_idx, n - 1)

        scaled_loc = scaler.transform(loc_df[FEATURE_COLS].values)
        targets = loc_df[LABEL_COL].values

        # Train sequences: target belongs to training slice.
        for i in range(TIME_STEPS, split_idx):
            X_train.append(scaled_loc[i - TIME_STEPS: i])
            y_train.append(targets[i])

        # Test sequences: target belongs to testing slice.
        for i in range(split_idx, n):
            X_test.append(scaled_loc[i - TIME_STEPS: i])
            y_test.append(targets[i])

    if not X_train or not X_test:
        raise ValueError("Insufficient data to create both train and test sequences.")

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
    )


def build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, len(FEATURE_COLS))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_and_print(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    print("\n--- Test Metrics ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(metrics["classification_report"])
    return metrics


def main():
    print(f"Loading + preprocessing dataset from {DATASET_PATH}...")
    df = load_and_preprocess_dataset(DATASET_PATH)

    # Fit scaler only on training feature rows (from each location).
    train_feature_rows = []
    for _, loc_df in df.groupby("Location", sort=False):
        loc_df = loc_df.sort_values("Date")
        n = len(loc_df)
        if n <= TIME_STEPS + 1:
            continue
        split_idx = max(TIME_STEPS + 1, int(n * TRAIN_RATIO))
        split_idx = min(split_idx, n - 1)
        train_feature_rows.append(loc_df.iloc[:split_idx][FEATURE_COLS].values)

    if not train_feature_rows:
        raise ValueError("Could not collect training rows for scaler fitting.")

    scaler = MinMaxScaler()
    scaler.fit(np.vstack(train_feature_rows))

    X_train, y_train, X_test, y_test = build_train_test_sequences(df, scaler)
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape : X={X_test.shape}, y={y_test.shape}")

    model = build_model()
    callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]

    print("Training LSTM...")
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating on test set...")
    y_prob = model.predict(X_test, verbose=0).flatten()
    metrics = evaluate_and_print(y_test.astype(int), y_prob)
    metrics["best_val_loss"] = float(min(history.history.get("val_loss", [float("nan")])))

    model.save(MODEL_PATH)
    with PREPROCESSOR_PATH.open("wb") as f:
        pickle.dump(
            {
                "feature_cols": FEATURE_COLS,
                "time_steps": TIME_STEPS,
                "scaler": scaler,
                "label_col": LABEL_COL,
            },
            f,
        )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved preprocessor to: {PREPROCESSOR_PATH}")
    print(f"Saved test metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
