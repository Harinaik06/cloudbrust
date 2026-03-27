import pickle
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model

MODEL_PATH = Path("lstm_model.h5")
PREPROCESSOR_PATH = Path("preprocessor.pkl")
DATASET_PATH = Path("weather.csv")


def load_preprocessor():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Missing preprocessor.pkl. Run train.py first.")

    with PREPROCESSOR_PATH.open("rb") as f:
        preprocessor = pickle.load(f)

    return (
        preprocessor["feature_cols"],
        preprocessor["time_steps"],
        preprocessor["scaler"],
        preprocessor["label_col"],
    )


FEATURE_COLS, TIME_STEPS, SCALER, LABEL_COL = load_preprocessor()
MODEL = load_model(MODEL_PATH)


def load_clean_dataset():
    required_cols = ["Date", "Location"] + FEATURE_COLS + [LABEL_COL]
    raw_df = pd.read_csv(DATASET_PATH)

    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    df = raw_df[required_cols].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Location"]).copy()

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)

    df[FEATURE_COLS] = (
        df.groupby("Location")[FEATURE_COLS]
        .transform(lambda g: g.interpolate(limit_direction="both"))
    )
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


DF_CLEAN = load_clean_dataset()
DATASET_LOCATIONS = sorted(DF_CLEAN["Location"].unique().tolist())

# Keep location-wise tables for sequence-safe lookups.
LOCATION_DATA = {
    loc: loc_df.sort_values("Date").reset_index(drop=True)
    for loc, loc_df in DF_CLEAN.groupby("Location")
}


def risk_and_label(confidence: float, profile: str = "historical"):
    # Use stricter thresholds for real-time/forecast to reduce false alarms.
    if profile in {"realtime", "forecast"}:
        if confidence >= 0.90:
            return "High Risk (Severe Warning)", "Cloudburst Warning"
        if confidence >= 0.75:
            return "Moderate Risk (Early Warning)", "Cloudburst Watch"
        return "Low Risk (Safe)", "No Cloudburst"

    if confidence >= 0.7:
        return "High Risk (Severe Warning)", "Cloudburst"
    if confidence >= 0.5:
        return "Moderate Risk (Early Warning)", "Cloudburst"
    return "Low Risk (Safe)", "No Cloudburst"


def build_trend_figure(humidity_series, temp_series, title_text, xaxis_title):
    days = list(range(1, TIME_STEPS + 1))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=days,
            y=humidity_series,
            name="Humidity 3pm (%)",
            mode="lines+markers",
            line=dict(color="#00d2ff", width=4, shape="spline"),
            marker=dict(size=10, symbol="circle", color="#00d2ff", line=dict(width=2, color="white")),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=temp_series,
            name="Max Temp (°C)",
            mode="lines+markers",
            line=dict(color="#ff2a2a", width=4, shape="spline"),
            marker=dict(size=10, symbol="square", color="#ff2a2a", line=dict(width=2, color="white")),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=title_text,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=xaxis_title, showgrid=True, gridcolor="rgba(255,255,255,0.1)", tickmode="linear")
    fig.update_yaxes(title_text="<b>Humidity (%)</b>", secondary_y=False, showgrid=True, gridcolor="rgba(255,255,255,0.1)", range=[0, 100])
    fig.update_yaxes(title_text="<b>Temperature (°C)</b>", secondary_y=True, showgrid=False)
    return fig


def set_scan_example(start_date: str, end_date: str):
    return start_date, end_date

# ── Single date prediction ──────────────────────────────────────────
def predict_by_date(date_str, location):
    try:
        if location not in LOCATION_DATA:
            return "Invalid location selected.", None

        target_date = pd.to_datetime(date_str, format="%d-%m-%Y")
        loc_df = LOCATION_DATA[location]
        max_hist_date = loc_df["Date"].max()
        dataset_min = DF_CLEAN["Date"].min().strftime("%d-%m-%Y")
        dataset_max = DF_CLEAN["Date"].max().strftime("%d-%m-%Y")

        # Future date branch: use live + forecast weather from Open-Meteo.
        if target_date > max_hist_date:
            # Try to find a matching city name in LOCATIONS_MAP (case-insensitive)
            matched_city = next(
                (city for city in LOCATIONS_MAP if city.lower() == location.lower()), None
            )
            if matched_city is None:
                supported = ", ".join(sorted(LOCATIONS_MAP.keys()))
                return (
                    f"Date {date_str} is beyond the historical dataset for {location}.\n"
                    f"Dataset covers: {dataset_min} to {dataset_max}\n\n"
                    f"Future-date forecast is supported for: {supported}.\n"
                    f"'{location}' is not in the future-forecast city list.",
                    None,
                )

            lat, lon = LOCATIONS_MAP[matched_city]
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&past_days={TIME_STEPS}&forecast_days=16"
                "&hourly=relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover"
                "&daily=temperature_2m_max,temperature_2m_min"
                "&timezone=Asia%2FKolkata"
            )
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()

            daily = data["daily"]
            hourly = data["hourly"]
            daily_dates = pd.to_datetime(daily["time"])

            target_idx_list = np.where(daily_dates == target_date.normalize())[0]
            if len(target_idx_list) == 0:
                min_f = daily_dates.min().strftime("%d-%m-%Y")
                max_f = daily_dates.max().strftime("%d-%m-%Y")
                return (
                    f"Future forecast unavailable for {target_date.strftime('%d-%m-%Y')}.\n"
                    f"Open-Meteo forecast coverage: {min_f} to {max_f}.\n"
                    f"Please enter a date within this window.",
                    None,
                )

            day_idx = int(target_idx_list[0])
            if day_idx < TIME_STEPS:
                return "Not enough weather context from API for this date. Please try a slightly later date.", None

            X_live = []
            for i in range(day_idx - TIME_STEPS, day_idx):
                idx_9am = i * 24 + 9
                idx_3pm = i * 24 + 15
                cloud9 = hourly["cloud_cover"][idx_9am]
                cloud3 = hourly["cloud_cover"][idx_3pm]

                X_live.append(
                    [
                        daily["temperature_2m_min"][i],
                        daily["temperature_2m_max"][i],
                        hourly["relative_humidity_2m"][idx_9am],
                        hourly["relative_humidity_2m"][idx_3pm],
                        hourly["surface_pressure"][idx_9am],
                        hourly["surface_pressure"][idx_3pm],
                        hourly["wind_speed_10m"][idx_9am],
                        hourly["wind_speed_10m"][idx_3pm],
                        (cloud9 / 12.5) if cloud9 is not None else 0,
                        (cloud3 / 12.5) if cloud3 is not None else 0,
                    ]
                )

            raw_df_live = pd.DataFrame(X_live, columns=FEATURE_COLS).ffill().bfill()
            raw_window = raw_df_live.values
            scaled_window = SCALER.transform(raw_window).reshape(1, TIME_STEPS, len(FEATURE_COLS))
            confidence = float(MODEL.predict(scaled_window, verbose=0)[0][0])
            risk_level, label = risk_and_label(confidence, profile="forecast")

            fig = build_trend_figure(
                raw_df_live["Humidity3pm"].values,
                raw_df_live["MaxTemp"].values,
                title_text=f"<b>Forecast Weather Trend for {location}</b>",
                xaxis_title="<b>Previous 10 Days Used for Forecast</b>",
            )
            output_text = (
                f"Date       : {target_date.strftime('%d-%m-%Y')} (Future Forecast)\n"
                f"Location   : {location}\n"
                f"Result     : {label}\n"
                f"Risk Level : {risk_level}\n"
                f"Confidence : {confidence:.3f}\n"
                "Actual     : Not available (future date)"
            )
            return output_text, fig

        matched = loc_df.index[loc_df["Date"] == target_date]
        if len(matched) == 0:
            nearest_idx = int((loc_df["Date"] - target_date).abs().idxmin())
            idx = nearest_idx
            used_date = loc_df.loc[idx, "Date"]
            note = (
                f"Requested date {date_str} was unavailable for {location}. "
                f"Using nearest historical date {used_date.strftime('%d-%m-%Y')}.\n"
            )
        else:
            idx = int(matched[0])
            note = ""

        if idx < TIME_STEPS:
            idx = TIME_STEPS
            adjusted_date = loc_df.loc[idx, "Date"].strftime("%d-%m-%Y")
            note += (
                f"Not enough prior history for the selected point. "
                f"Using earliest prediction-capable date {adjusted_date}.\n"
            )

        raw_window = loc_df.loc[idx - TIME_STEPS: idx - 1, FEATURE_COLS].values
        scaled_window = SCALER.transform(raw_window).reshape(1, TIME_STEPS, len(FEATURE_COLS))
        confidence = float(MODEL.predict(scaled_window, verbose=0)[0][0])
        risk_level, label = risk_and_label(confidence, profile="historical")
        actual = int(loc_df.loc[idx, LABEL_COL])

        fig = build_trend_figure(
            loc_df.loc[idx - TIME_STEPS: idx - 1, "Humidity3pm"].values,
            loc_df.loc[idx - TIME_STEPS: idx - 1, "MaxTemp"].values,
            title_text=f"<b>Historical Weather Trend for {location}</b>",
            xaxis_title="<b>Previous 10 Days Used for Prediction</b>",
        )

        used_date_str = loc_df.loc[idx, "Date"].strftime("%d-%m-%Y")
        output_text = (
            f"{note}"
            f"Date       : {used_date_str}\n"
            f"Location   : {location}\n"
            f"Result     : {label}\n"
            f"Risk Level : {risk_level}\n"
            f"Confidence : {confidence:.3f}\n"
            f"Actual     : {'Cloudburst (Rain)' if actual == 1 else 'No Cloudburst'}"
        )
        return output_text, fig
    except ValueError:
        return "Invalid format. Use DD-MM-YYYY (e.g. 15-06-2012).", None
    except Exception as e:
        return f"Error: {str(e)}", None


# ── Scan a date range for all predicted cloudbursts ─────────────────
def scan_range(start_str, end_str):
    try:
        start = pd.to_datetime(start_str, format="%d-%m-%Y")
        end   = pd.to_datetime(end_str,   format="%d-%m-%Y")

        if start > end:
            return "Start date must be before End date.", None

        dataset_min = DF_CLEAN["Date"].min()
        dataset_max = DF_CLEAN["Date"].max()

        # If the entire range is in the future, redirect the user
        if start > dataset_max:
            return (
                f"The date range {start_str} – {end_str} is entirely beyond the historical dataset.\n"
                f"Dataset covers: {dataset_min.strftime('%d-%m-%Y')} to {dataset_max.strftime('%d-%m-%Y')}\n\n"
                f"For future dates, please use the 'Single Date Prediction' tab, which supports "
                f"live Open-Meteo forecast for the next 16 days.",
                None,
            )

        # Clip end date to dataset max and warn if partially future
        clipped = False
        if end > dataset_max:
            end = dataset_max
            clipped = True

        subset = DF_CLEAN[(DF_CLEAN["Date"] >= start) & (DF_CLEAN["Date"] <= end)]
        if subset.empty:
            return (
                f"No historical data found between {start_str} and {end_str}.\n"
                f"Dataset covers: {dataset_min.strftime('%d-%m-%Y')} to {dataset_max.strftime('%d-%m-%Y')}",
                None,
            )

        rows_for_prediction = []
        batch_X = []
        for loc, loc_df in LOCATION_DATA.items():
            in_range_idx = loc_df.index[(loc_df["Date"] >= start) & (loc_df["Date"] <= end)].tolist()
            for idx in in_range_idx:
                if idx < TIME_STEPS:
                    continue
                raw_window = loc_df.loc[idx - TIME_STEPS: idx - 1, FEATURE_COLS].values
                batch_X.append(SCALER.transform(raw_window))
                rows_for_prediction.append((loc, idx))

        if not batch_X:
            return "Not enough preceding data for any date in this range.", None

        batch_X = np.array(batch_X, dtype=np.float32)
        preds = MODEL.predict(batch_X, verbose=0).flatten()

        cloudburst_rows = []
        dates = []
        confidences = []

        for i, (loc, idx) in enumerate(rows_for_prediction):
            loc_df = LOCATION_DATA[loc]
            d_obj = loc_df.loc[idx, "Date"]
            dates.append(d_obj)
            confidences.append(preds[i])

            if preds[i] > 0.5:
                d = d_obj.strftime("%d-%m-%Y")
                conf = preds[i]
                risk = "High Risk" if conf >= 0.7 else "Moderate Risk"
                actual = int(loc_df.loc[idx, LABEL_COL])
                cloudburst_rows.append(
                    f"{d}  |  {loc:<15}  |  Risk: {risk:<13}  |  Confidence: {conf:.3f}  |  Actual: {'Rain' if actual==1 else 'No Rain'}"
                )

        fig = go.Figure()
        fig.add_hrect(y0=0.7, y1=1.05, fillcolor="red", opacity=0.15, line_width=0, layer="below")
        fig.add_hrect(y0=0.5, y1=0.7, fillcolor="orange", opacity=0.15, line_width=0, layer="below")
        fig.add_hrect(y0=0.0, y1=0.5, fillcolor="green", opacity=0.05, line_width=0, layer="below")
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk (70%)", annotation_font_color="white")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Moderate Risk (50%)", annotation_font_color="white")

        fig.add_trace(go.Scatter(
            x=dates, y=confidences, mode='lines+markers', name='Confidence Score',
            line=dict(color='#00ff9d', width=4, shape='spline'),
            marker=dict(size=10, color='#00ff9d', line=dict(width=2, color='white'))
        ))

        fig.update_layout(
            title=f"<b>Prediction Confidence Trends</b>",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", font=dict(color="white"),
            yaxis=dict(title="<b>Probability of Cloudburst</b>", range=[0, 1.05], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(title="<b>Date</b>", showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )

        clipped_note = (
            f"\n⚠️ End date clipped to dataset max ({dataset_max.strftime('%d-%m-%Y')}). "
            "Use 'Single Date Prediction' for future dates."
        ) if clipped else ""

        if not cloudburst_rows:
            return f"No Cloudburst predicted between {start_str} and {end_str}.{clipped_note}", fig

        header = f"Cloudburst predictions ({len(cloudburst_rows)} found) between {start_str} and {end_str}:{clipped_note}\n"
        header += "-" * 90 + "\n"
        return header + "\n".join(cloudburst_rows), fig

    except ValueError:
        return "Invalid date format. Use DD-MM-YYYY.", None
    except Exception as e:
        return f"Error: {str(e)}", None


# ── Present Date Prediction using Open-Meteo API ────────────────────
LOCATIONS_MAP = {
    'Mumbai': (19.0760, 72.8777),
    'Delhi': (28.7041, 77.1025),
    'Chennai': (13.0827, 80.2707),
    'Kolkata': (22.5726, 88.3639),
    'Bengaluru': (12.9716, 77.5946),
    'Hyderabad': (17.3850, 78.4867),
    'Pune': (18.5204, 73.8567),
    'Ahmedabad': (23.0225, 72.5714),
    'Surat': (21.1702, 72.8311),
    'Lucknow': (26.8467, 80.9462),
    'Hyderabad': (17.3850, 78.4867)
}

def _extract_day_features(daily, hourly, day_idx):
    """Extract feature vector for a single day index from Open-Meteo data."""
    idx_9am = day_idx * 24 + 9
    idx_3pm = day_idx * 24 + 15
    cloud9 = hourly['cloud_cover'][idx_9am]
    cloud3 = hourly['cloud_cover'][idx_3pm]
    return [
        daily['temperature_2m_min'][day_idx],
        daily['temperature_2m_max'][day_idx],
        hourly['relative_humidity_2m'][idx_9am],
        hourly['relative_humidity_2m'][idx_3pm],
        hourly['surface_pressure'][idx_9am],
        hourly['surface_pressure'][idx_3pm],
        hourly['wind_speed_10m'][idx_9am],
        hourly['wind_speed_10m'][idx_3pm],
        (cloud9 / 12.5) if cloud9 is not None else 0,
        (cloud3 / 12.5) if cloud3 is not None else 0,
    ]


def predict_present(location_name):
    try:
        if location_name not in LOCATIONS_MAP:
            return "Please select a valid location.", None, None, ""

        lat, lon = LOCATIONS_MAP[location_name]
        # Fetch past 10 days + 7 future days so we can slide windows over forecast days
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&past_days=10&forecast_days=7"
            "&hourly=relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover"
            "&daily=temperature_2m_max,temperature_2m_min"
            "&timezone=Asia%2FKolkata"
        )
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()

        daily = data['daily']
        hourly = data['hourly']
        daily_dates = pd.to_datetime(daily['time'])
        total_days = len(daily['time'])

        # Build full feature matrix for all available days
        all_features = []
        for d in range(total_days):
            all_features.append(_extract_day_features(daily, hourly, d))
        all_df = pd.DataFrame(all_features, columns=FEATURE_COLS).ffill().bfill()

        # ── Present date prediction (index = 10, the 11th day = today) ──
        present_idx = 10  # past_days=10 means index 10 is today
        raw_window = all_df.iloc[present_idx - TIME_STEPS: present_idx].values
        scaled_window = SCALER.transform(raw_window).reshape(1, TIME_STEPS, len(FEATURE_COLS))
        confidence_today = float(MODEL.predict(scaled_window, verbose=0)[0][0])
        risk_level, label = risk_and_label(confidence_today, profile="realtime")
        today_date = pd.Timestamp.now().strftime("%d-%m-%Y")

        # ── Past 10 days trend chart ──
        past_humidity = all_df.iloc[:present_idx]["Humidity3pm"].values
        past_temp = all_df.iloc[:present_idx]["MaxTemp"].values
        days_axis = list(range(1, present_idx + 1))

        fig_past = make_subplots(specs=[[{"secondary_y": True}]])
        fig_past.add_trace(
            go.Scatter(x=days_axis, y=past_humidity, name="Humidity 3pm (%)",
                       mode='lines+markers', line=dict(color='#00d2ff', width=4, shape='spline'),
                       marker=dict(size=10, symbol='circle', color='#00d2ff', line=dict(width=2, color='white'))),
            secondary_y=False,
        )
        fig_past.add_trace(
            go.Scatter(x=days_axis, y=past_temp, name="Max Temp (°C)",
                       mode='lines+markers', line=dict(color='#ff2a2a', width=4, shape='spline'),
                       marker=dict(size=10, symbol='square', color='#ff2a2a', line=dict(width=2, color='white'))),
            secondary_y=True,
        )
        fig_past.update_layout(
            title=f"<b>Live Weather Trends for {location_name}</b>",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_past.update_xaxes(title_text="<b>Past 10 Days</b>", showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickmode='linear')
        fig_past.update_yaxes(title_text="<b>Humidity (%)</b>", secondary_y=False, showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 100])
        fig_past.update_yaxes(title_text="<b>Temperature (°C)</b>", secondary_y=True, showgrid=False)

        # ── 7-day future forecast predictions ──
        forecast_dates = []
        forecast_confidences = []
        forecast_labels = []
        forecast_risks = []

        for future_offset in range(1, 8):  # next 7 days
            target_idx = present_idx + future_offset
            if target_idx >= total_days:
                break
            window_start = target_idx - TIME_STEPS
            if window_start < 0:
                continue
            raw_w = all_df.iloc[window_start:target_idx].values
            scaled_w = SCALER.transform(raw_w).reshape(1, TIME_STEPS, len(FEATURE_COLS))
            conf = float(MODEL.predict(scaled_w, verbose=0)[0][0])
            rl, lbl = risk_and_label(conf, profile="forecast")
            forecast_dates.append(daily_dates[target_idx].strftime("%d-%m-%Y"))
            forecast_confidences.append(conf)
            forecast_labels.append(lbl)
            forecast_risks.append(rl)

        # ── 7-day forecast chart ──
        marker_colors = [
            "#ff4444" if c >= 0.90 else "#ffaa00" if c >= 0.75 else "#00ff9d"
            for c in forecast_confidences
        ]
        fig_forecast = go.Figure()
        fig_forecast.add_hrect(y0=0.90, y1=1.05, fillcolor="red", opacity=0.12, line_width=0, layer="below")
        fig_forecast.add_hrect(y0=0.75, y1=0.90, fillcolor="orange", opacity=0.12, line_width=0, layer="below")
        fig_forecast.add_hrect(y0=0.0, y1=0.75, fillcolor="green", opacity=0.05, line_width=0, layer="below")
        fig_forecast.add_hline(y=0.90, line_dash="dash", line_color="red",
                               annotation_text="High Risk (90%)", annotation_font_color="white")
        fig_forecast.add_hline(y=0.75, line_dash="dash", line_color="orange",
                               annotation_text="Moderate Risk (75%)", annotation_font_color="white")
        fig_forecast.add_trace(go.Bar(
            x=forecast_dates,
            y=forecast_confidences,
            name="Cloudburst Probability",
            marker_color=marker_colors,
            marker_line=dict(width=1, color="white"),
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>",
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_confidences,
            mode="lines+markers",
            name="Trend",
            line=dict(color="white", width=2, dash="dot"),
            marker=dict(size=8, color="white"),
        ))
        fig_forecast.update_layout(
            title=f"<b>7-Day Cloudburst Forecast for {location_name}</b>",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            font=dict(color="white"),
            yaxis=dict(title="<b>Cloudburst Probability</b>", range=[0, 1.05],
                       showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(title="<b>Forecast Date</b>", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # ── Text outputs ──
        output_text = (
            f"Prediction : PRESENT DATE ({today_date})\n"
            f"Location   : {location_name}\n"
            f"Result     : {label}\n"
            f"Risk Level : {risk_level}\n"
            f"Confidence : {confidence_today:.3f}"
        )

        forecast_lines = [f"7-Day Cloudburst Forecast for {location_name}:"]
        forecast_lines.append("-" * 65)
        for i, d in enumerate(forecast_dates):
            emoji = "🔴" if forecast_confidences[i] >= 0.90 else "🟡" if forecast_confidences[i] >= 0.75 else "🟢"
            forecast_lines.append(
                f"{emoji}  {d}  |  {forecast_labels[i]:<20}  |  {forecast_risks[i]:<30}  |  Conf: {forecast_confidences[i]:.3f}"
            )
        forecast_text = "\n".join(forecast_lines)

        return output_text, fig_past, fig_forecast, forecast_text
    except Exception as e:
        return f"Error fetching live prediction: {str(e)}", None, None, ""


# ── Gradio UI ────────────────────────────────────────────────────────
def build_interface():
    with gr.Blocks(title="Cloudburst Prediction using LSTM") as interface:
        gr.Markdown("# Indian Cloudburst & Extreme Rainfall Prediction")

        with gr.Tab("Real-Time (Present) Prediction"):
            with gr.Row():
                loc_input = gr.Dropdown(choices=list(LOCATIONS_MAP.keys()), label="Select City", value="Mumbai")
            rt_btn = gr.Button("Predict Present Weather + 7-Day Forecast", variant="primary")
            with gr.Row():
                rt_output = gr.Textbox(label="Today's Prediction", lines=6)
                rt_plot = gr.Plot(label="Past 10-Day Weather Trends")
            gr.Markdown("### 7-Day Future Cloudburst Forecast")
            with gr.Row():
                rt_forecast_text = gr.Textbox(label="7-Day Forecast Summary", lines=10)
                rt_forecast_plot = gr.Plot(label="7-Day Forecast Confidence Chart")
            rt_btn.click(
                fn=predict_present,
                inputs=loc_input,
                outputs=[rt_output, rt_plot, rt_forecast_plot, rt_forecast_text],
            )

        with gr.Tab("Scan Date Range"):
            gr.Markdown("Scan a range of dates and list all predicted cloudburst days.")
            with gr.Row():
                start_input = gr.Textbox(label="Start Date (DD-MM-YYYY)", placeholder="e.g. 01-01-2025")
                end_input   = gr.Textbox(label="End Date (DD-MM-YYYY)",   placeholder="e.g. 31-12-2025")
            with gr.Row():
                ex_2022_btn = gr.Button("Use Example: 2022 Full Year")
                ex_2025_btn = gr.Button("Use Example: 2025 Full Year")
            with gr.Row():
                scan_output = gr.Textbox(label="Cloudburst Dates Found", lines=15)
                scan_plot = gr.Plot(label="Prediction Confidence Trend")
            scan_btn = gr.Button("Scan for Cloudbursts", variant="primary")
            scan_btn.click(fn=scan_range, inputs=[start_input, end_input], outputs=[scan_output, scan_plot])
            ex_2022_btn.click(
                fn=lambda: set_scan_example("01-01-2022", "31-12-2022"),
                inputs=[],
                outputs=[start_input, end_input],
            )
            ex_2025_btn.click(
                fn=lambda: set_scan_example("01-01-2025", "31-12-2025"),
                inputs=[],
                outputs=[start_input, end_input],
            )

            gr.Examples(
                examples=[["01-01-2022", "31-12-2022"], ["01-01-2025", "31-12-2025"]],
                inputs=[start_input, end_input],
            )

        with gr.Tab("Single Date Prediction"):
            gr.Markdown("Enter a specific date and select a location to check if cloudburst is predicted.")
            with gr.Row():
                date_input = gr.Textbox(label="Date (DD-MM-YYYY)", placeholder="e.g. 25-03-2026")
                loc_dropdown = gr.Dropdown(choices=DATASET_LOCATIONS, label="Select Location", value=DATASET_LOCATIONS[0])
            with gr.Row():
                single_output = gr.Textbox(label="Result", lines=6)
                single_plot = gr.Plot(label="Weather Trends")
            predict_btn = gr.Button("Predict", variant="primary")
            predict_btn.click(fn=predict_by_date, inputs=[date_input, loc_dropdown], outputs=[single_output, single_plot])

            # Show 3 real cloudburst dates as examples (with location).
            real_cb = DF_CLEAN[DF_CLEAN[LABEL_COL] == 1][["Date", "Location"]].head(3)
            gr.Examples(
                examples=[[row["Date"].strftime("%d-%m-%Y"), row["Location"]] for _, row in real_cb.iterrows()],
                inputs=[date_input, loc_dropdown],
            )

    return interface


if __name__ == "__main__":
    build_interface().launch(
        server_name="0.0.0.0",
        share=True,
        css="""
        footer {display: none !important;}
        .built-with, .gradio-container .prose a[href*='gradio.app'] {display: none !important;}
        """
    )
