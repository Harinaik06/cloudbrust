# Cloudburst Project Run Guide

Use these commands from the project folder.

## 1) Preprocess Dataset

```powershell
.\.venv\Scripts\python.exe preprocess.py
```

Output:
- `weather_cleaned.csv`

## 2) Train + Test LSTM

```powershell
.\.venv\Scripts\python.exe train_test_lstm.py
```

Outputs:
- `lstm_model.h5`
- `preprocessor.pkl`
- `test_metrics.json`

## 3) Run Gradio App (Localhost Only)

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:7860` (localhost).

## 4) Run Gradio App (Network-Accessible)

```powershell
.\.venv\Scripts\python.exe app.py
```

1. Terminal shows `http://0.0.0.0:7860` → Find your IPv4: `ipconfig` (look for "IPv4 Address", e.g., 192.168.1.100).
2. On other laptops (same WiFi): Open `http://192.168.1.100:7860`.
3. Windows Firewall: Allow Python/port 7860 if blocked.

## Files for Demo

- `preprocess.py` - data cleaning and preprocessing
- `train_test_lstm.py` - model training and testing with metrics
- `app.py` - user interface for prediction
