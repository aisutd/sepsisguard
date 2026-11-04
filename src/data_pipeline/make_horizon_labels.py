"""
make_horizon_labels.py

Builds earlier-warning labels from SepsisLabel.
PhysioNet sets SepsisLabel = 1 from 6 hours before sepsis onset, so
onset = first hour labeled 1 + 6. For a horizon H, the new label is 1 from
(onset - H) onward. H = 6 rebuilds the original label; H = 12 and 24 warn earlier.
Reads the existing train/test files, so the patient split does not change.
Run from the project root: python src/data_pipeline/make_horizon_labels.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")
LABEL = "SepsisLabel"
PHYSIONET_SHIFT = 6         # SepsisLabel turns 1 this many hours before onset
HORIZONS = [6, 12, 24]      # how many hours before onset each new label turns 1
                            # each becomes a col. label_6h...

def find_onsets(df):
    # One row per patient: patient_id, site, onset_hour (empty if never septic).
    sites = df.groupby("patient_id")["site"].first()
    # First hour each septic patient is labeled 1 (non-septic patients aren't in here).
    first_positive = df[df[LABEL] == 1].groupby("patient_id")["ICULOS"].min()
    onsets = sites.to_frame()
    # map() leaves non-septic patients as NaN; Int64 keeps whole numbers and
    # saves NaN as an empty cell.
    onsets["onset_hour"] = (onsets.index.map(first_positive) + PHYSIONET_SHIFT).astype("Int64")
    return onsets.reset_index()


def add_horizon_labels(df, onsets):
    # Look up each row's onset hour from its patient_id.
    onset = df["patient_id"].map(onsets.set_index("patient_id")["onset_hour"]).astype(float)
    for h in HORIZONS:
        # 1 from (onset - h) onward. For non-septic patients onset is NaN,
        # the comparison is False, so they stay 0.
        df[f"label_{h}h"] = (df["ICULOS"] >= onset - h).astype(int)
    return df


def report(df, name):
    # Counts for the notes. Every septic patient is positive at every horizon;
    # what changes is how many hours are labeled 1, and how many patients
    # are already 1 at their first hour (no warning window left for the model).
    first_hours = df.groupby("patient_id").head(1)
    print(f"\n{name}: {df['patient_id'].nunique()} patients")
    for h in HORIZONS:
        col = f"label_{h}h"
        patients = int(df.groupby("patient_id")[col].max().sum())
        print(f"  H={h:>2}: {patients} patients labeled 1, {df[col].mean():.2%} of hourly rows, "
              f"{int(first_hours[col].sum())} already 1 at their first hour")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # Onsets for every patient (train + test) go in one small file.
    onsets = find_onsets(pd.concat([train, test]))
    onsets.to_csv(DATA_DIR / "onsets.csv", index=False)
    print(f"Saved onsets.csv: {onsets['onset_hour'].notna().sum()} septic patients of {len(onsets)}")

    for name, df in [("train", train), ("test", test)]:
        df = add_horizon_labels(df, onsets)
        # Check: H = 6 must rebuild the original label exactly.
        mismatches = int((df["label_6h"] != df[LABEL]).sum())
        print(f"{name}: label_6h vs SepsisLabel mismatches = {mismatches}")
        assert mismatches == 0, "H = 6 does not match SepsisLabel, check the onset maths"
        report(df, name)
        df.to_csv(DATA_DIR / f"{name}_labeled.csv", index=False)

    print(f"\nSaved train_labeled.csv and test_labeled.csv to {DATA_DIR}/")


if __name__ == "__main__":
    main()
