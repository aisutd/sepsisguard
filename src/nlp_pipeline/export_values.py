"""Export each patient's top 5 SHAP features together with their real values.

The summaries in Week 4 quote feature values ("Lactate (1.4) raised ..."), so
the NLP side needs the value next to each SHAP number. The shared export
(data/sample/shap_top_features.json, src/baseline_shap/export_shap.py) has no
values, so this script builds the same file with a "value" added. It leaves
that export and its script alone. When the 12 h export with values lands, point
verify_summary.py at it with --input and this script is no longer needed.

Inputs (both already in the repo, nothing is downloaded):
  models/xgb_baseline.json            the trained baseline model
  data/processed/sample_patients.csv  the latest hour of 50 real test patients

Output:
  data/sample/shap_top_features_with_values.json
    [{"patient_id": "p000006",
      "top_features": [{"feature": "ICULOS", "shap_value": 0.0171, "value": 19}, ...]}, ...]

SHAP is computed the same way as export_shap.py (shap.Explainer on the model's
predicted probability, with the sample itself as the background), so the
numbers are on the same scale as the shared export. Each value and its SHAP
value come from the same row.

Note on "value": it is the number the model was given, after cleaning. The
cleaning step carries the last measurement forward and fills hours with no
measurement yet using the training median, so a value is the model's input and
not always a fresh measurement.

Needs xgboost and shap (see requirements.txt); the rest of the NLP pipeline
does not. Run from the repo root:
    python src/nlp_pipeline/export_values.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models/xgb_baseline.json"
SAMPLE_PATH = REPO_ROOT / "data/processed/sample_patients.csv"
OUTPUT_PATH = REPO_ROOT / "data/sample/shap_top_features_with_values.json"

TOP_K = 5


def main():
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    sample = pd.read_csv(SAMPLE_PATH)

    # Use the model's own column order, so a column in the CSV that the model
    # never saw (patient_id, site, SepsisLabel) can't slip in as a feature.
    X = sample[model.feature_names]

    def predict(data):
        return model.predict(xgb.DMatrix(pd.DataFrame(data, columns=X.columns)))

    # seed makes the permutation sampling repeatable, so reruns give the same file.
    explainer = shap.Explainer(predict, X, seed=0)
    shap_values = explainer(X).values  # one row per patient, one column per feature

    records = []
    for i, patient_id in enumerate(sample["patient_id"]):
        # Strongest features first, by absolute SHAP value.
        order = np.argsort(-np.abs(shap_values[i]))[:TOP_K]
        records.append({
            "patient_id": str(patient_id),
            "top_features": [
                {
                    "feature": X.columns[j],
                    "shap_value": round(float(shap_values[i][j]), 4),
                    "value": round(float(X.iloc[i, j]), 2),
                }
                for j in order
            ],
        })

    OUTPUT_PATH.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {len(records)} patients to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
