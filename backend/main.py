# SepsisGuard backend: a small FastAPI server that sends data to the dashboard.
# Run from the backend/ folder:  uvicorn main:app --reload
# Then open http://127.0.0.1:8000/docs to try each endpoint in the browser.

import json                      # reads the sample patients file
from pathlib import Path         # builds file paths that work on any computer

from fastapi import FastAPI 
from confidence import score_confidence
app= FastAPI()

# where the sample patients file lives (made by src/data_pipeline/export_sample.py).
# Built from this file's location (backend/ -> sepsisguard/ -> data/sample/),
# so it works no matter which folder the server is started from.
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "patients_sample.json"

# Takes one patient row (vitals, labs, demographics) as JSON and says how
# trustworthy it is, based on how many fields are missing (high / medium / low).
@app.post("/confidence")
def confidence(row:dict):
    return score_confidence(row)

# Simple check that the server is running.
@app.get("/health")
def health():
    return {"status": "ok"}

#returns the 20 sample patients: each one's latest heart rate,
# respiratory rate, temperature, whether they developed sepsis,
# and their data-confidence label.
@app.get("/patients")
def patients():
    with open(SAMPLE_PATH) as f:
        return json.load(f)