"""
clean_patients.py

Cleaning pipeline for the PhysioNet 2019 sepsis data.
Saves plain CSV files (only needs pandas and numpy).

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")       # contains training_setA/ and training_setB/
OUT_DIR = Path("data/processed")  # where the cleaned files are saved
TEST_SIZE = 0.2   # 20% of patients go to the test set
SEED = 42         # fixed seed so the split is the same every run

# Data contract: the columns (and order) of the cleaned output
ID_COLS = ["patient_id", "site", "ICULOS"]  # ICULOS = hours since ICU admission
FEATURES = [
    # vitals
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    # labs
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "Hct", "Hgb",
    "PTT", "WBC", "Platelets",
    # demographics
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
]
LABEL = "SepsisLabel"   # 1 = septic at this hour, 0 = not
OUTPUT_COLS = ID_COLS + FEATURES + [LABEL]

# Yes/no columns
BINARY = ["Gender", "Unit1", "Unit2"]

# 1. Load the files (site A and site B combined into one dataframe)
def load_patients(limit=None):

    # Find all patient files in each site's folder, in a fixed (sorted) order.
    files_a = sorted((DATA_DIR / "training_setA").glob("*.psv"))
    files_b = sorted((DATA_DIR / "training_setB").glob("*.psv"))
    if not files_a or not files_b:
        sys.exit(f"No .psv files found in {DATA_DIR}/training_setA or training_setB. "
                 "Run this from the project root.")

    # limit: if given, only load this many patients (python clean_patients 20)
    if limit:
        # Quick test: take half from each site so both hospitals are in the sample.
        files_a = files_a[: limit // 2]
        files_b = files_b[: limit - limit // 2]

    print(f"Loading {len(files_a)} files from site A and {len(files_b)} from site B...")

    # read each file
    frames = []     # empty list that will collect one small table per patient 
    for site, files in [("A", files_a), ("B", files_b)]:
        for path in files:
            df = pd.read_csv(path, sep="|")   # .psv files are pipe-separated
            df["patient_id"] = path.stem      # e.g. "p000001"
            df["site"] = site                 # which hospital the patient came from
            frames.append(df)                   # add patient's table to list

    # Stack every patient's rows into one table (1,552,210).
    return pd.concat(frames, ignore_index=True)

# 3. One row per hour for every patient
def fill_skipped_hours(df):
 
    # If an hour shows up twice, keep the last one.
    df = df.drop_duplicates(subset=["patient_id", "ICULOS"], keep="last").copy()
    df["ICULOS"] = df["ICULOS"].astype(int)

    # For each patient, list every hour from their first to their last.
    hours = df.groupby("patient_id")["ICULOS"].agg(["min", "max"])
    hours["ICULOS"] = [list(range(lo, hi + 1)) for lo, hi in zip(hours["min"], hours["max"])]
    # explode() turns each patient's list of hours into one row per hour,
    # giving a "grid" table with columns patient_id and ICULOS.
    grid = hours["ICULOS"].explode().reset_index()
    grid["ICULOS"] = grid["ICULOS"].astype(int)

    # Join the real data onto the full hour list. Skipped hours become empty
    # rows, which the forward-fill step fills in next.
    print(f"Adding {len(grid) - len(df)} rows for skipped hours")
    # [OUTPUT_COLS] puts the columns back in the data-contract order.
    return grid.merge(df, on=["patient_id", "ICULOS"], how="left")[OUTPUT_COLS]


# 5. Split by patient 
def split_by_patient(df):
    # Splitting by patient means no patient's hours end up in both sets,
    # so the model is never tested on someone it has already seen.
    # Each patient is "septic" if they are ever labeled 1.
    is_septic = df.groupby("patient_id")[LABEL].max()

    # Split septic and non-septic patients 80/20 separately,
    # so both sets keep the same sepsis rate.
    rng = np.random.default_rng(SEED)
    test_ids = []
    for label in [0, 1]:
        ids = is_septic[is_septic == label].index.to_numpy()
        rng.shuffle(ids)  # random order; same order every run because of SEED
        # The first 20% of the shuffled patients go to test; the rest go to train.
        test_ids.extend(ids[: round(len(ids) * TEST_SIZE)])

    # True for every row that belongs to a test patient; ~ flips it for train.
    in_test = df["patient_id"].isin(test_ids)
    return df[~in_test].copy(), df[in_test].copy()


# 7. Checks: sepsis split and Site A vs Site B
def report_sepsis_split(df, name):
    """Print how many patients (and hourly rows) are sepsis-positive."""
    # Patient level: share of patients who are ever septic (should be ~7%).
    # Row level: share of hourly rows labeled 1 (much lower, because septic
    # patients are labeled 0 for most of their stay).
    per_patient = df.groupby("patient_id")[LABEL].max()
    print(f"{name:<6} {len(per_patient):>6} patients, {int(per_patient.sum())} septic "
          f"({per_patient.mean():.1%} of patients, {df[LABEL].mean():.1%} of hourly rows)")


def compare_sites(df):
    #Print and save a side-by-side summary of each feature at site A vs site B.
    # Median for measurements, mean (share of patients) for yes/no columns.
    numeric = [c for c in FEATURES if c not in BINARY]
    # .T (transpose) makes each feature a row, with columns "A" and "B".
    summary = pd.concat([
        df.groupby("site")[numeric].median().T,
        df.groupby("site")[BINARY].mean().T,
    ])

    # Flag measurements whose site medians differ by more than 20%.
    # pct_diff = gap between A and B as a % of their average. It is NaN when
    # both are 0 (e.g. BaseExcess), which just means "no difference".
    a, b = summary["A"], summary["B"]
    summary["pct_diff"] = ((a - b).abs() / ((a.abs() + b.abs()) / 2) * 100).round(1)
    flagged = [c for c in numeric if summary.loc[c, "pct_diff"] > 20]

    print("\nSite A vs Site B after cleaning (median; share of patients for yes/no columns):")
    print(summary.round(2).to_string())
    print(f"Measurements differing by more than 20%: {flagged or 'none'}")
    summary.to_csv(OUT_DIR / "site_comparison.csv")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. Load A + B and keep only the contract columns.
    df = load_patients(limit)
    # Stop early with a clear message if any expected column is missing.
    missing =[c for c in OUTPUT_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Raw data is missing expected columns: {missing}")
    df = df[OUTPUT_COLS].copy()
    print(f"Loaded {len(df)} rows for {df['patient_id'].nunique()} patients")

    # 3. One row per hour.
    df = fill_skipped_hours(df)

    # 4. Forward-fill within each patient. A lab value stays the patient's
    #    current value until it is measured again. This only uses earlier
    #    hours (never the future) and groupby keeps it within one patient.
    #    Rows must be in time order for "carry the last value forward" to work.
    #    site and SepsisLabel are filled too, so rows added for skipped hours
    #    get them (SepsisLabel only goes 0 -> 1, so carrying it forward is safe).
    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    fill_cols = ["site"] + FEATURES + [LABEL]
    df[fill_cols] = df.groupby("patient_id")[fill_cols].ffill()

    # 5. Split into train and test (before the median fill, see step 6).
    train, test = split_by_patient(df)

    # 6. Fill what's still missing (hours before a value was ever measured)
    #    with the TRAINING median, so test data never leaks into training.
    medians = train[FEATURES].median()  # one median per column
    train[FEATURES] = train[FEATURES].fillna(medians)
    test[FEATURES] = test[FEATURES].fillna(medians)

    # 7. Checks: sepsis split (~7%) and Site A vs Site B after cleaning.
    print("\nSepsis split:")
    report_sepsis_split(pd.concat([train, test]), "All")
    report_sepsis_split(train, "Train")
    report_sepsis_split(test, "Test")
    compare_sites(pd.concat([train, test]))

    # Sanity checks before saving: stop instead of saving bad data.
    assert not train[FEATURES].isna().any().any(), "train still has missing values"
    assert not test[FEATURES].isna().any().any(), "test still has missing values"
    assert list(train.columns) == OUTPUT_COLS and list(test.columns) == OUTPUT_COLS
    assert not set(train["patient_id"]) & set(test["patient_id"]), "a patient is in both sets"

    # 8. Save as CSV. index=False leaves out pandas' row numbers.
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    # Small sample for the frontend (PatientList.jsx): the latest hour for
    # 50 test patients. The full files are too big to load in a browser.
    # tail(1) = each patient's last row; head(50) = the first 50 of those.
    test.groupby("patient_id").tail(1).head(50).to_csv(OUT_DIR / "sample_patients.csv", index=False)

    print(f"\nSaved train.csv, test.csv, sample_patients.csv and site_comparison.csv to {OUT_DIR}/")


# Only run main() when this file is run directly, not when it is imported.
if __name__ == "__main__":
    main()