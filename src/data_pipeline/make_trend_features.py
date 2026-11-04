"""
make_trend_features.py

Adds trend columns: how much each key vital changed over the last 6 and 12 hours.
Uses only earlier hours of the same patient, never the future.
Run after make_horizon_labels.py: python src/data_pipeline/make_trend_features.py
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")
TREND_VITALS = ["HR", "Resp", "Temp", "MAP", "O2Sat"]  # MAP = mean blood pressure
WINDOWS = [6, 12]                                     # hours to look back


def add_trends(df):
    # Rows must be in time order within each patient for "hours ago" to be right.
    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    by_patient = df.groupby("patient_id")
    for col in TREND_VITALS:
        for w in WINDOWS:
            # clean_patients.py made one row per hour, so diff(w) = value now
            # minus value w hours ago. groupby keeps it inside one patient.
            # The first w hours of a stay have nothing to compare to, so they
            # get 0 ("no change seen yet").
            df[f"{col}_change_{w}h"] = by_patient[col].diff(w).fillna(0)
    return df


def main():
    for name in ["train", "test"]:
        df = pd.read_csv(DATA_DIR / f"{name}_labeled.csv")
        df = add_trends(df)
        df.to_csv(DATA_DIR / f"{name}_features.csv", index=False)
        print(f"Saved {name}_features.csv ({len(df)} rows)")

    # Print the new column names to post in Discord.
    new_cols = [f"label_{h}h" for h in [6, 12, 24]] + \
               [f"{c}_change_{w}h" for c in TREND_VITALS for w in WINDOWS]
    print("\nNew columns:", ", ".join(new_cols))


if __name__ == "__main__":
    main()
