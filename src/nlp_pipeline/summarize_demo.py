"""Write the checked summaries for the 8 demo patients (Final Sprint Round 2, issue #52).

Output: data/sample/summaries_demo.json, plus a copy in frontend/public/ for
SummaryCard.jsx. One entry per demo patient, in demo_patients.json order:

    {"patient_id": "p003455",
     "icu_hour": 113,                     # the patient's snapshot_icu_hour
     "horizon": 12,
     "source": "XGBoost baseline",
     "explanation_file": "computed by summarize_demo.py",
     "top_features": [{"feature_name": "ICULOS", "shap_value": 0.03, "value": 113}, ...],
     "summary": "...",                    # exactly what DistilBART wrote
     "verified": true,
     "failure_reason": null,              # why it failed, when verified is false
     "failure_types": [],
     "fallback_summary": "..."}           # the template text, always safe to show

"summary" is always the AI-written text, even when it failed, so the failure
stays visible. Anything shown to a reader should be "summary" when verified is
true and "fallback_summary" otherwise. The backend matches entries by
patient_id and icu_hour (backend/main.py, find_at_snapshot), which is why
icu_hour is stored.

Where the reasons come from
---------------------------
The demo contract says every component must describe a patient at their
snapshot_icu_hour. For each patient this script uses, in order:

  1. An entry in data/sample/explanations_demo.json (Immaculate's export) with
     the same patient_id AND an "icu_hour" equal to the snapshot hour.
  2. Otherwise, SHAP computed here, the way the shared export does it: the 12 h
     XGBoost baseline (models/xgb_h12.json), shap.Explainer on its predicted
     probability. The patient's row is rebuilt from demo_patients.json
     ("inputs_at_snapshot", which holds all inputs except ICULOS, and
     ICULOS = the snapshot hour, since ICULOS counts hours in the ICU). It
     matches the row in data/processed/test.csv, so no big data file is needed.
     The background is the 50 test patients in data/processed/sample_patients.csv,
     the same rows the Week 4 run used.

When this was written, explanations_demo.json held 10 rows for 2 patients
(p000011, p000006) who are not demo patients, without icu_hour or values, so
all 8 patients used route 2. Once the shared file is fixed, rerunning this
script picks it up with no code change.

Direction: SHAP values are signed (positive raised the predicted risk,
negative lowered it), so the "raised/lowered" wording and the direction check
stay on. An unsigned source, such as a plain importance ranking from the TFT,
would make them meaningless; this script refuses such a source instead of
guessing (see check_signed).

Needs torch + transformers (DistilBART) and xgboost + shap (route 2), all in
the repo .venv. On a Mac without Homebrew's libomp, xgboost can borrow the copy
that scikit-learn ships:
    export DYLD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/sklearn/.dylibs
Run from the repo root:
    python src/nlp_pipeline/summarize_demo.py
"""

import json
import shutil
import sys
from pathlib import Path

from generate_summary import TOP_K, feature_column, generate_summaries, generate_summary
from verify_summary import check_planted_errors, summarize_and_verify

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "data/sample"
PATIENTS_PATH = SAMPLE_DIR / "demo_patients.json"
EXPLANATIONS_PATH = SAMPLE_DIR / "explanations_demo.json"
MODEL_PATH = REPO_ROOT / "models/xgb_h12.json"
BACKGROUND_PATH = REPO_ROOT / "data/processed/sample_patients.csv"
OUTPUT_PATH = SAMPLE_DIR / "summaries_demo.json"
FRONTEND_COPY = REPO_ROOT / "frontend/public/summaries_demo.json"

HORIZON = 12
SOURCE = "XGBoost baseline"
BACKEND = "distilbart"


def shared_explanations(patients):
    """Entries from explanations_demo.json that match a demo patient at their snapshot hour.

    Returns {patient_id: entry}. Entries for other patients or other hours are
    ignored, as the backend does, because the screen must not mix hours.
    """
    if not EXPLANATIONS_PATH.exists():
        return {}
    snapshot = {p["patient_id"]: p["snapshot_icu_hour"] for p in patients}
    found = {}
    for row in json.loads(EXPLANATIONS_PATH.read_text()):
        pid = row.get("patient_id")
        if pid in snapshot and row.get("icu_hour") == snapshot[pid]:
            found[pid] = row
    return found


def computed_explanations(patients):
    """Top 5 SHAP features at the snapshot hour for each patient, from the 12 h model.

    Returns {patient_id: top_features list}. The heavy imports live here so the
    script still runs without xgboost when the shared file covers everyone.
    """
    import numpy as np
    import pandas as pd
    import shap
    import xgboost as xgb

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    columns = model.feature_names  # the model's own order; nothing else can slip in

    rows = []
    for p in patients:
        row = dict(p["inputs_at_snapshot"])
        row["ICULOS"] = p["snapshot_icu_hour"]
        missing = set(columns) - row.keys()
        if missing:
            raise ValueError(f"{p['patient_id']} is missing model inputs: {sorted(missing)}")
        rows.append(row)
    X = pd.DataFrame(rows)[columns]
    background = pd.read_csv(BACKGROUND_PATH)[columns]

    def predict(data):
        return model.predict(xgb.DMatrix(pd.DataFrame(data, columns=columns)))

    # seed makes the permutation sampling repeatable, so reruns give the same file.
    shap_values = shap.Explainer(predict, background, seed=0)(X).values

    result = {}
    for i, p in enumerate(patients):
        order = np.argsort(-np.abs(shap_values[i]))[:TOP_K]  # strongest first
        result[p["patient_id"]] = [
            {
                "feature_name": columns[j],
                "shap_value": round(float(shap_values[i][j]), 4),
                "value": round(float(X.iloc[i, j]), 2),
            }
            for j in order
        ]
    return result


def check_signed(entry):
    """Refuse a source with no direction (every value >= 0), see the module docstring."""
    values = [f["shap_value"] for f in entry["top_features"]]
    if len(values) > 1 and all(v >= 0 for v in values) and entry.get("source") not in (None, SOURCE):
        raise ValueError(
            f"{entry['patient_id']}: '{entry.get('source')}' gives only positive values, so it may "
            "be unsigned. Turn off the raised/lowered wording before summarizing it."
        )


def build_entries(patients):
    """One explanation entry per demo patient, at their snapshot hour."""
    shared = shared_explanations(patients)
    missing = [p for p in patients if p["patient_id"] not in shared]
    computed = computed_explanations(missing) if missing else {}

    entries = []
    for p in patients:
        pid = p["patient_id"]
        if pid in shared:
            row = shared[pid]
            features = [
                {
                    "feature_name": feature_column(f),
                    "shap_value": f["shap_value"],
                    # The shared file may leave out values; the contract has them.
                    "value": f.get("value", p["inputs_at_snapshot"].get(feature_column(f))),
                }
                for f in row["top_features"]
            ]
            source, origin = row.get("source", SOURCE), EXPLANATIONS_PATH.name
        else:
            features, source, origin = computed[pid], SOURCE, "computed by summarize_demo.py"
        entry = {
            "patient_id": pid,
            "icu_hour": p["snapshot_icu_hour"],
            "horizon": HORIZON,
            "source": source,
            "explanation_file": origin,
            "top_features": features,
        }
        check_signed(entry)
        entries.append(entry)
    return entries, len(shared)


def main():
    patients = json.loads(PATIENTS_PATH.read_text())
    entries, n_shared = build_entries(patients)
    print(f"Reasons: {n_shared} from {EXPLANATIONS_PATH.name}, {len(entries) - n_shared} computed here")

    summaries = generate_summaries(entries, BACKEND)
    records = []
    for entry, summary in zip(entries, summaries):
        check = summarize_and_verify(entry, BACKEND, summary)
        records.append({
            **entry,
            "summary": check["summary"],
            "verified": check["verified"],
            # Same keys on every entry, so the card never has to guess.
            "failure_reason": check.get("failure_reason"),
            "failure_types": check.get("failure_types", []),
            "fallback_summary": check.get("fallback_summary") or generate_summary(entry, "template"),
            "warnings": check.get("warnings", []),
        })

    passed = sum(r["verified"] for r in records)
    print(f"{BACKEND}: {passed}/{len(records)} demo summaries passed")
    for r in records:
        status = "PASS" if r["verified"] else f"FAIL ({r['failure_reason']})"
        print(f"\n  {r['patient_id']} @ hour {r['icu_hour']}: {status}\n    {r['summary']}")

    planted = check_planted_errors(entries)
    print()
    for kind, (caught, total) in planted.items():
        print(f"Deliberately wrong demo summaries, {kind}: {caught}/{total} caught")

    OUTPUT_PATH.write_text(json.dumps(records, indent=2) + "\n")
    shutil.copyfile(OUTPUT_PATH, FRONTEND_COPY)
    print(f"\nWrote {len(records)} records to {OUTPUT_PATH} and {FRONTEND_COPY}")


if __name__ == "__main__":
    sys.exit(main())
