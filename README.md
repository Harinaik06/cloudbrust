# Weather Prediction LSTM App

## Overview
This project is a machine learning application for weather prediction using an LSTM (Long Short-Term Memory) neural network. It includes data preprocessing, model training/testing, and a web interface for predictions.

## Files
- `app.py`: Gradio web app for weather predictions.
- `train_test_lstm.py`: LSTM model training and testing script.
- `preprocess.py`: Data preprocessing utilities.
- `lstm_model.h5`: Trained Keras LSTM model.
- `preprocessor.pkl`: Pickled preprocessor (e.g., scaler).
- `weather.csv`: Dataset.
- `requirements.txt`: Python dependencies.
- `RUN_GUIDE.md`: Detailed run instructions.

## Quick Start

1. **Setup Environment**
   ```
   python -m venv .venv
   .venv\\Scripts\\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```
   python app.py
   ```
   Open the local Gradio URL (usually http://127.0.0.1:7860).

3. **Train/Test Model** (if needed)
   ```
   python train_test_lstm.py
   ```

## Dependencies
See `requirements.txt` (e.g., tensorflow, gradio, pandas, scikit-learn).

## Notes
- Model trained on `weather.csv`.
- Uses GPU if available for training.

For detailed guide, see `RUN_GUIDE.md`.
