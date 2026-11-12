"""
build_demo_files.py

Builds the shared demo data that every part of the final screen reads:
    data/sample/demo_patients.json  - who the 8 demo patients are
    data/sample/demo_scores.json    - their TFT risk scores (6h / 12h / 24h)

Run from the project root (sepsisguard/):
    python src/demo/build_demo_files.py            # build both files
    python src/demo/build_demo_files.py --ids-only # just choose and print the 8 IDs

Inputs:
    data/processed/test.csv                  cleaned test set (clean_patients.py)
    data/raw/training_setA|B/<id>.psv        raw files, to tell measured from filled-in values
    data/processed/tft_scores_{6h,12h,24h}.csv   Arnav's full test-set scores (from Drive)

How the demo patients are chosen (fixed rule, decided BEFORE looking at any model output):
    1. Start from every patient in test.csv.
    2. Keep patients the TFT can score: 25+ hourly rows (24-hour history + 1 hour to predict,
       the same cut-off train_tft.py uses).
    3. Split them into "developed sepsis" (SepsisLabel is 1 at any hour) and "did not".
    4. From each group, sort by patient_id and draw 4 at random with random_state=42
       (the team's usual seed). Scores are never used to pick patients.

How each patient's snapshot hour is chosen:
    The ICU hour with that patient's highest 12h TFT score (earliest hour if tied).

Both decisions are saved ("locked") in data/sample/demo_selection.json the first time they
are made. Later runs read the lock file instead of choosing again, so the demo patients and
their snapshot hours never change, even if the models are retrained.
To deliberately start over, delete demo_selection.json (and tell the team, everyone uses these IDs).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
SAMPLE = PROJECT_ROOT / "data" / "sample"

TEST_PATH = PROCESSED / "test.csv"
LOCK_PATH = SAMPLE / "demo_selection.json"
PATIENTS_OUT = SAMPLE / "demo_patients.json"
SCORES_OUT = SAMPLE / "demo_scores.json"

HORIZONS = ["6h", "12h", "24h"]
SNAPSHOT_HORIZON = "12h"   # the snapshot hour is the peak of this horizon's score
N_PER_GROUP = 4            # 4 septic + 4 non-septic = 8 demo patients
MIN_HOURS = 25             # TFT needs 24 hours of history + 1 hour to predict
SEED = 42

# Reuse the backend's confidence rule (backend/confidence.py) so the demo header shows the
# same high / medium / low label the /confidence endpoint would give for this hour.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from confidence import EXPECTED_FIELDS, score_confidence  # noqa: E402


def show(path):
    """Path for printing: relative to the project root when possible (shorter to read)."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


# 1. Choose the 8 patients (by the fixed rule; no model scores involved)
def choose_patients(test):
    # One row per patient: site, how many hours they have, and whether they ever became septic.
    per_patient = test.groupby("patient_id").agg(
        site=("site", "first"),
        hours=("ICULOS", "nunique"),
        developed_sepsis=("SepsisLabel", "max"),
    ).reset_index()

    eligible = per_patient[per_patient["hours"] >= MIN_HOURS].sort_values("patient_id")

    chosen = []
    # Septic group first, then non-septic, so the order in the files is predictable.
    for label in [1, 0]:
        group = eligible[eligible["developed_sepsis"] == label]
        picked = group.sample(n=N_PER_GROUP, random_state=SEED).sort_values("patient_id")
        chosen.append(picked)

    chosen = pd.concat(chosen, ignore_index=True)
    print(f"Eligible patients (>= {MIN_HOURS} hours): {len(eligible)} "
          f"({int(eligible['developed_sepsis'].sum())} developed sepsis)")
    return [
        {"patient_id": r.patient_id, "site": r.site, "developed_sepsis": bool(r.developed_sepsis)}
        for r in chosen.itertuples()
    ]


# 2. Load the TFT score files, keeping only our 8 patients
def load_scores(patient_ids):
    scores = {}
    for h in HORIZONS:
        path = PROCESSED / f"tft_scores_{h}.csv"
        if not path.exists():
            sys.exit(f"Missing {show(path)}. Download Arnav's "
                     f"tft_scores_{h}.csv from Drive into data/processed/ and run again.")
        df = pd.read_csv(path)
        df = df[df["patient_id"].isin(patient_ids)]
        scores[h] = df[["patient_id", "ICULOS", "risk_score"]].rename(columns={"risk_score": h})

    # One table: patient_id, ICULOS, 6h, 12h, 24h. Inner join keeps only hours all 3 models scored.
    merged = scores["6h"]
    for h in HORIZONS[1:]:
        merged = merged.merge(scores[h], on=["patient_id", "ICULOS"], how="inner", validate="one_to_one")

    missing = set(patient_ids) - set(merged["patient_id"])
    if missing:
        sys.exit(f"The TFT did not score these demo patients: {sorted(missing)}. "
                 "This should not happen for patients with 25+ hours; check the score files.")
    return merged.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)


# 3. Snapshot hour = hour of the highest 12h score
def peak_hour(patient_scores):
    # idxmax returns the first (earliest) row when there is a tie.
    return int(patient_scores.loc[patient_scores[SNAPSHOT_HORIZON].idxmax(), "ICULOS"])


# 4. Imputed share: how much of the snapshot hour was filled in rather than measured
def raw_row(patient_id, site, icu_hour):
    """The patient's raw (uncleaned) values at this ICU hour, as a dict.
    Returns {} if the hour does not exist in the raw file (an hour the cleaning step
    added), in which case every input at that hour was filled in."""
    path = RAW / f"training_set{site}" / f"{patient_id}.psv"
    raw = pd.read_csv(path, sep="|")
    rows = raw[raw["ICULOS"] == icu_hour]
    return {} if rows.empty else rows.iloc[-1].to_dict()  # last copy, like clean_patients.py


def imputed_info(patient_id, site, icu_hour):
    # A value counts as "filled in" when it is blank in the raw file at this hour.
    # In the cleaned file it was then carried forward from an earlier hour or set to the
    # training median, so the model saw a number that was not measured at this hour.
    # score_confidence counts exactly these blanks across the same 35 model inputs.
    result = score_confidence(raw_row(patient_id, site, icu_hour))
    return {
        "imputed_share": result["missing_fraction"],   # 0.0 = all measured, 1.0 = all filled in
        "imputed_count": result["missing_count"],
        "total_inputs": result["total_fields"],
        "imputed_fields": result["missing_fields"],
        "confidence": result["confidence"],
    }


# Lock file: choose once, reuse forever
def load_or_create_lock(test, need_snapshots):
    lock = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else None

    if lock is None:
        lock = {
            "rule": (f"test.csv patients with >= {MIN_HOURS} hourly rows; {N_PER_GROUP} who developed "
                     f"sepsis and {N_PER_GROUP} who did not, each drawn with pandas .sample("
                     f"random_state={SEED}) from patients sorted by patient_id. "
                     f"snapshot_icu_hour = hour of highest {SNAPSHOT_HORIZON} TFT score."),
            "patients": choose_patients(test),
        }
        print(f"Chose the demo patients and saved them to {show(LOCK_PATH)}")
    else:
        print(f"Using the locked demo patients in {show(LOCK_PATH)}")

    ids = [p["patient_id"] for p in lock["patients"]]
    scores = None
    if need_snapshots:
        scores = load_scores(ids)
        # Only fill in snapshot hours that have not been set yet; never overwrite one.
        for p in lock["patients"]:
            if "snapshot_icu_hour" not in p:
                p["snapshot_icu_hour"] = peak_hour(scores[scores["patient_id"] == p["patient_id"]])

    SAMPLE.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n")
    return lock, scores


# Build the two shared files
def build(lock, scores, test):
    patients_out, scores_out = [], []

    for p in lock["patients"]:
        pid, site, hour = p["patient_id"], p["site"], p["snapshot_icu_hour"]
        hist = test[test["patient_id"] == pid]
        snap = hist[hist["ICULOS"] == hour].iloc[0]
        s = scores[scores["patient_id"] == pid]
        at_snap = s[s["ICULOS"] == hour].iloc[0]

        patients_out.append({
            "patient_id": pid,
            "site": site,
            "snapshot_icu_hour": hour,
            "total_icu_hours": int(hist["ICULOS"].max()),
            # Outcome is shown on screen as "demo only": a real ICU would not know it yet.
            "developed_sepsis": p["developed_sepsis"],
            **imputed_info(pid, site, hour),
            # The cleaned values the model saw at the snapshot hour (rounded for display).
            "inputs_at_snapshot": {f: round(float(snap[f]), 2) for f in EXPECTED_FIELDS},
        })

        scores_out.append({
            "patient_id": pid,
            "snapshot_icu_hour": hour,
            # Scores at the snapshot hour. Relative risk for ranking, not calibrated probabilities.
            "scores": {h: round(float(at_snap[h]), 4) for h in HORIZONS},
            # Every hour up to and including the snapshot, for a trend line. Hours after
            # the snapshot are left out so the screen never shows the "future".
            "history": [
                {"icu_hour": int(r["ICULOS"]), **{h: round(float(r[h]), 4) for h in HORIZONS}}
                for r in s[s["ICULOS"] <= hour].to_dict("records")
            ],
        })

    PATIENTS_OUT.write_text(json.dumps(patients_out, indent=2) + "\n")
    SCORES_OUT.write_text(json.dumps(scores_out, indent=2) + "\n")
    print(f"Saved {show(PATIENTS_OUT)} and {show(SCORES_OUT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids-only", action="store_true",
                        help="only choose (or read) the 8 patients and print them; no score files needed")
    args = parser.parse_args()

    test = pd.read_csv(TEST_PATH, usecols=["patient_id", "site", "ICULOS", "SepsisLabel", *EXPECTED_FIELDS])
    lock, scores = load_or_create_lock(test, need_snapshots=not args.ids_only)

    print("\nDemo patients:")
    for p in lock["patients"]:
        outcome = "developed sepsis" if p["developed_sepsis"] else "no sepsis"
        snap = p.get("snapshot_icu_hour", "not set yet (needs score files)")
        print(f"  {p['patient_id']}  site {p['site']}  {outcome:<16}  snapshot hour: {snap}")

    if not args.ids_only:
        build(lock, scores, test)


if __name__ == "__main__":
    main()
