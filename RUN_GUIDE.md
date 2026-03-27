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

## 3) Run Gradio App

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open the local URL shown in terminal (for example `http://127.0.0.1:7866`).

## Files for Demo

- `preprocess.py` - data cleaning and preprocessing
- `train_test_lstm.py` - model training and testing with metrics
- `app.py` - user interface for prediction
