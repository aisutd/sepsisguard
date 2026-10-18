# Get 20 real patients from test.csv and saves their latest vitals, sepsis outcome 
# and data-confidence score to data/sample/patients_sample.json.

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # the sepsisguard/ folder
sys.path.append(str(ROOT / "backend"))              # so we can import confidence.py
from confidence import score_confidence

TEST_CSV = ROOT / "data" / "processed" / "test.csv"     # input file
OUT_JSON = ROOT / "data" / "sample" / "patients_sample.json"        #output file

df = pd.read_csv(TEST_CSV)  # read data into a DataFrame

# Latest hour for each patient = the row with the biggest ICULOS
latest = df.sort_values(["patient_id", "ICULOS"]).groupby("patient_id").tail(1).set_index("patient_id")
# Did the patient ever get SepsisLabel = 1?
latest["developed_sepsis"] = df.groupby("patient_id")["SepsisLabel"].max()

# 10 sepsis + 10 non-sepsis; random_state makes it pick the same patients every run
picked = pd.concat([
    latest[latest["developed_sepsis"] == 1].sample(10, random_state=42),
    latest[latest["developed_sepsis"] == 0].sample(10, random_state=42),
])

# build output records
records = []
for patient_id, row in picked.iterrows():
    conf = score_confidence(row.to_dict())          # Harish's function on a real row
    records.append({
        "patient_id": patient_id,
        "site": row["site"],
        "hours_in_icu": int(row["ICULOS"]),
        "heart_rate": row["HR"],
        "resp_rate": row["Resp"],
        "temperature": row["Temp"],
        "developed_sepsis": bool(row["developed_sepsis"]),
        "confidence": conf["confidence"],
        "missing_count": conf["missing_count"],
    })

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
# to_json turns NaN into null, which plain json.dumps would not do
pd.DataFrame(records).to_json(OUT_JSON, orient="records", indent=2)
print(f"Saved {len(records)} patients to {OUT_JSON}")
