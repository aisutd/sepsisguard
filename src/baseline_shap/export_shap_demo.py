"""Export SHAP reasons for the 8 demo patients at their snapshot hours (Final Sprint Round 2).

Output: data/sample/shap_demo.json, one entry per demo patient in demo_patients.json order:

    {"patient_id": "p003455",
     "icu_hour": 113,                  # the patient's snapshot_icu_hour
     "horizon": 12,
     "source": "XGBoost baseline",
     "top_features": [{"feature_name": "ICULOS", "shap_value": 0.3921, "value": 113.0}, ...]}

Why this file exists: the backend (backend/main.py, reasons_for) searches every
data/sample/*shap*.json file for an entry with the patient's ID AND an "icu_hour"
equal to their snapshot hour. The older SHAP files only cover non-demo patients
and have no icu_hour, so the reasons panel showed "not available" for everyone.

Method: the same as summarize_demo.py (Anikait), so the reasons panel and the
written summary agree:
  - model: the 12h XGBoost baseline, models/xgb_h12.json
  - the patient's row at the snapshot hour is rebuilt from demo_patients.json
    ("inputs_at_snapshot" holds every input except ICULOS; ICULOS = the
    snapshot hour, since it counts hours in the ICU)
  - background: the 50 test patients in data/processed/sample_patients.csv
  - shap.Explainer with seed=0, so reruns give the same file

Run from sepsisguard/:
    python src/baseline_shap/export_shap_demo.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
PATIENTS_PATH = REPO_ROOT / "data/sample/demo_patients.json"
MODEL_PATH = REPO_ROOT / "models/xgb_h12.json"
BACKGROUND_PATH = REPO_ROOT / "data/processed/sample_patients.csv"
# The name must contain "shap" so the backend finds it.
OUTPUT_PATH = REPO_ROOT / "data/sample/shap_demo.json"

HORIZON = 12
TOP_K = 5
SOURCE = "XGBoost baseline"


def rows_at_snapshot(patients, columns):
    """One input row per patient at their snapshot hour, in the model's column order."""
    rows = []
    for p in patients:
        row = dict(p["inputs_at_snapshot"])
        row["ICULOS"] = p["snapshot_icu_hour"]
        # Stop with a clear message instead of letting SHAP fail on a missing input
        missing = set(columns) - row.keys()
        if missing:
            raise ValueError(f"{p['patient_id']} is missing model inputs: {sorted(missing)}")
        rows.append(row)
    return pd.DataFrame(rows)[columns]


def main():
    patients = json.loads(PATIENTS_PATH.read_text())

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    columns = model.feature_names  # use the model's own inputs, in its own order

    X = rows_at_snapshot(patients, columns)
    background = pd.read_csv(BACKGROUND_PATH)[columns]

    def predict(data):
        """Predicted sepsis risk for rows SHAP passes in."""
        return model.predict(xgb.DMatrix(pd.DataFrame(data, columns=columns)))

    print(f"Computing SHAP for {len(X)} demo patients with {MODEL_PATH.name}...")
    shap_values = shap.Explainer(predict, background, seed=0)(X).values

    entries = []
    for i, p in enumerate(patients):
        # Strongest effect first, whether it raised (+) or lowered (-) the risk
        order = np.argsort(-np.abs(shap_values[i]))[:TOP_K]
        entries.append({
            "patient_id": p["patient_id"],
            "icu_hour": p["snapshot_icu_hour"],
            "horizon": HORIZON,
            "source": SOURCE,
            "top_features": [
                {
                    "feature_name": columns[j],
                    "shap_value": round(float(shap_values[i][j]), 4),
                    "value": round(float(X.iloc[i, j]), 2),  # the patient's measured value
                }
                for j in order
            ],
        })

    OUTPUT_PATH.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {len(entries)} entries to {OUTPUT_PATH}")
    for e in entries:
        names = ", ".join(f["feature_name"] for f in e["top_features"])
        print(f"  {e['patient_id']} @ hour {e['icu_hour']}: {names}")


if __name__ == "__main__":
    main()
