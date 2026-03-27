from pathlib import Path

import pandas as pd

FEATURE_COLS = [
    "MinTemp",
    "MaxTemp",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Cloud9am",
    "Cloud3pm",
]
LABEL_COL = "RainTomorrow"
REQUIRED_COLS = ["Date", "Location"] + FEATURE_COLS + [LABEL_COL]

DEFAULT_INPUT = Path("weather.csv")
DEFAULT_OUTPUT = Path("weather_cleaned.csv")


def load_and_preprocess_dataset(csv_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)

    missing_cols = [c for c in REQUIRED_COLS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = raw_df[REQUIRED_COLS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Location"]).copy()
    df["Location"] = df["Location"].astype(str)

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Keep each city's time order intact.
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)

    # Fill missing feature values location-wise.
    df[FEATURE_COLS] = (
        df.groupby("Location")[FEATURE_COLS]
        .transform(lambda g: g.interpolate(limit_direction="both"))
    )
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


def main():
    print(f"Loading raw dataset: {DEFAULT_INPUT}")
    df = load_and_preprocess_dataset(DEFAULT_INPUT)
    print(f"Cleaned rows: {len(df)}")
    print(f"Locations: {df['Location'].nunique()}")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

    df.to_csv(DEFAULT_OUTPUT, index=False)
    print(f"Saved cleaned dataset to: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
