# Cloudburst Prediction System

[![Gradio](https://img.shields.io/badge/Gradio-UI-EB5C33)](http://127.0.0.1:7860)

Indian Cloudburst & Extreme Rainfall prediction using LSTM neural network on weather data. Supports historical analysis and live Open-Meteo forecasts for 11 cities.

## 🎯 Features
- **Historical predictions** on dataset (Sydney weather, adaptable)
- **Live forecasts** for Mumbai, Delhi, Chennai, Kolkata, Bengaluru, **Hyderabad**, Pune, Ahmedabad, Surat, Lucknow
- Interactive Plotly charts (weather trends, confidence bars)
- Risk levels: High/Moderate/Low Cloudburst probability
- Gradio UI: Real-time, Date range scan, Single date tabs

Demo: http://127.0.0.1:7860

## 📊 Model Performance
- Test Accuracy: 77.5%
- Precision: 50.1%, Recall: 8.2% (imbalanced dataset)

## 🛠 Setup

1. **Virtual Environment**:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Preprocess Dataset**:
   ```
   python preprocess.py
   ```

4. **Train LSTM Model**:
   ```
   python train_test_lstm.py
   ```

5. **Launch UI**:
   ```
   python app.py
   ```

## 📁 Files
| File | Purpose |
|------|---------|
| `app.py` | Gradio UI + Open-Meteo API forecasts |
| `preprocess.py` | Clean weather.csv → weather_cleaned.csv |
| `train_test_lstm.py` | Train LSTM(64→32), save model/preprocessor |
| `lstm_model.h5` | Trained model |
| `weather.csv` | Raw dataset |
| `requirements.txt` | Dependencies |

## 🚀 Usage Examples
- **Real-time**: Select city → Get today + 7-day forecast
- **Scan Range**: e.g. 01-01-2022 to 31-12-2022 for historical cloudbursts
- **Single Date**: e.g. 01-01-2026 Hyderabad → Live forecast (no historical fallback)

## 🤝 GitHub Workflow
```
git init && git remote add origin https://github.com/Harinaik06/project.git
git add . && git commit -m "Update"
git push
```

## Improvements
- Add Indian weather dataset
- Improve recall with class balancing
- Docker deployment

**Author**: Harinaik06 | License: MIT

