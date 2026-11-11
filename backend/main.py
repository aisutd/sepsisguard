import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Lets "from confidence import ..." work whether the server is started from
# the project root or from inside backend/
sys.path.append(str(Path(__file__).resolve().parent))
from confidence import label_from_share, score_confidence  # noqa: E402

app = FastAPI(title="SepsisGuard backend")

# CORS: a browser blocks a page on one address (the React dashboard, port 5173)
# from calling a server on another address (this backend, port 8000) unless the
# server says it's allowed. This allows the dashboard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# All shared files live in data/sample/. The path is built from this file's
# location (backend/ -> sepsisguard/ -> data/sample/), so it works from any folder.
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
PATIENTS_FILE = SAMPLE_DIR / "demo_patients.json"        # Maritza: the 8 demo patients
SCORES_FILE = SAMPLE_DIR / "demo_scores.json"            # Maritza: TFT scores by hour
THRESHOLDS_FILE = SAMPLE_DIR / "alert_thresholds.json"   # Harish: TFT alert cutoffs

# The alert rule checks the horizons in this order. The first score at or
# above its cutoff sets the level: 6h = Urgent, else 12h = High risk,
# else 24h = Elevated risk, else No alert.
HORIZONS = ["6h", "12h", "24h"]
ALERT_LEVELS = {"6h": "Urgent", "12h": "High risk", "24h": "Elevated risk"}
NO_ALERT = "No alert"

SCORE_NOTE = ("Relative risk for ranking patients, not probabilities. "
              "Do not display as percentages.")
OUTCOME_NOTE = "Demo only: a real ICU would not know the outcome yet."


def read_json(path):
    """Read one shared file. Returns None if the file isn't there yet."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def by_patient(rows):
    """Turn a list of per-patient entries into {patient_id: entry} for quick lookup."""
    return {row["patient_id"]: row for row in rows or []}


def load_cutoffs():
    """The alert cutoffs from alert_thresholds.json, or None if the file is missing."""
    data = read_json(THRESHOLDS_FILE)
    return data["thresholds"] if data else None


def alert_level(scores, cutoffs):
    """Apply the alert rule to one set of scores ({"6h": ..., "12h": ..., "24h": ...}).
    Returns (level, the horizon that triggered it or None)."""
    for horizon in HORIZONS:
        score = scores.get(horizon)
        if score is not None and score >= cutoffs[horizon]:
            return ALERT_LEVELS[horizon], horizon
    return NO_ALERT, None


def find_at_snapshot(pattern, patient_id, snapshot_hour):
    """Search every data/sample file whose name matches `pattern` (e.g. '*shap*.json')
    for an entry about this patient AT their snapshot hour. The demo contract says
    everything shown must describe the patient at that hour, so entries without a
    matching "icu_hour" are skipped. Returns (entry, file name) or (None, None)."""
    for path in sorted(SAMPLE_DIR.glob(pattern)):
        rows = read_json(path)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if row.get("patient_id") == patient_id and row.get("icu_hour") == snapshot_hour:
                return row, path.name
    return None, None


def confidence_for(patient):
    """Confidence label from the share of inputs that were filled in, via confidence.py."""
    share = patient.get("imputed_share")
    if share is None:
        return {"available": False, "message": "Confidence not available: no imputed_share for this patient."}
    return {
        "available": True,
        "label": label_from_share(share),
        "imputed_share": share,
        "imputed_count": patient.get("imputed_count"),
        "total_inputs": patient.get("total_inputs"),
        "imputed_fields": patient.get("imputed_fields", []),
    }


def reasons_for(patient_id, snapshot_hour):
    """Top SHAP features (Immaculate's files) at the snapshot hour."""
    row, source = find_at_snapshot("*shap*.json", patient_id, snapshot_hour)
    if row is None:
        return {"available": False,
                "message": f"Reasons not available yet for {patient_id} at ICU hour {snapshot_hour}."}
    # Older SHAP files call the name "feature", newer ones "feature_name"; accept both
    features = [{"feature_name": f.get("feature_name", f.get("feature")),
                 "shap_value": f.get("shap_value"),
                 "value": f.get("value")}
                for f in row.get("top_features", [])]
    return {"available": True, "source": source, "icu_hour": snapshot_hour,
            "horizon": row.get("horizon"), "top_features": features}


def summary_for(patient_id, snapshot_hour):
    """Written summary (Anikait's files) at the snapshot hour."""
    row, source = find_at_snapshot("*summar*.json", patient_id, snapshot_hour)
    if row is None:
        return {"available": False,
                "message": f"Summary not available yet for {patient_id} at ICU hour {snapshot_hour}."}
    return {"available": True, "source": source, "icu_hour": snapshot_hour,
            "text": row.get("summary"), "verified": row.get("verified")}


def load_patients():
    """The demo patient list, or a clear 503 error if the file is missing."""
    patients = read_json(PATIENTS_FILE)
    if patients is None:
        raise HTTPException(status_code=503,
                            detail="Patients not available: data/sample/demo_patients.json is missing.")
    return patients

@app.get("/health")
def health():
    """Simple check that the server is running."""
    return {"status": "ok"}


@app.post("/confidence")
def confidence(row: dict):
    """Takes one raw patient row (vitals, labs, demographics) and says how complete it is."""
    return score_confidence(row)


@app.get("/patients")
def list_patients():
    """The 8 demo patients, each with their alert level at their snapshot hour."""
    patients = load_patients()
    scores = by_patient(read_json(SCORES_FILE))
    cutoffs = load_cutoffs()

    result = []
    for p in patients:
        score_row = scores.get(p["patient_id"])
        if score_row is None or cutoffs is None:
            level = "not available"
        else:
            level, _ = alert_level(score_row["scores"], cutoffs)
        share = p.get("imputed_share")
        result.append({
            "patient_id": p["patient_id"],
            "site": p.get("site"),
            "snapshot_icu_hour": p["snapshot_icu_hour"],
            "alert_level": level,
            "confidence": label_from_share(share) if share is not None else "not available",
            "outcome": {"developed_sepsis": p.get("developed_sepsis"), "note": OUTCOME_NOTE},
        })
    return result


@app.get("/patients/{patient_id}")
def patient_detail(patient_id: str):
    """Everything about one patient at their snapshot hour: scores (and their history
    up to that hour), alert level, confidence, reasons, summary and outcome."""
    patients = load_patients()
    patient = by_patient(patients).get(patient_id)
    if patient is None:
        known = ", ".join(p["patient_id"] for p in patients)
        raise HTTPException(status_code=404,
                            detail=f"Patient {patient_id} not available. Demo patients: {known}")

    snapshot = patient["snapshot_icu_hour"]
    score_row = by_patient(read_json(SCORES_FILE)).get(patient_id)
    cutoffs = load_cutoffs()

    # Alert level at the snapshot hour
    if score_row is None:
        alert = {"level": "not available", "message": "Scores not available: missing from demo_scores.json."}
    elif cutoffs is None:
        alert = {"level": "not available", "message": "Cutoffs not available: alert_thresholds.json is missing."}
    else:
        level, triggered_by = alert_level(score_row["scores"], cutoffs)
        alert = {"level": level, "triggered_by": triggered_by, "cutoffs": cutoffs,
                 "rule": "6h at or above its cutoff = Urgent, else 12h = High risk, "
                         "else 24h = Elevated risk, else No alert"}

    # Scores at the snapshot, plus hour-by-hour history for trend lines.
    # Only hours up to the snapshot are included, so the screen never shows the future.
    if score_row is None:
        scores = {"available": False, "message": "Scores not available: missing from demo_scores.json."}
    else:
        history = []
        for hour in score_row.get("history", []):
            if hour["icu_hour"] > snapshot:
                continue
            entry = dict(hour)
            entry["alert_level"] = alert_level(hour, cutoffs)[0] if cutoffs else "not available"
            history.append(entry)
        scores = {"available": True, "note": SCORE_NOTE,
                  "at_snapshot": score_row["scores"], "history": history}

    return {
        "patient_id": patient_id,
        "site": patient.get("site"),
        "snapshot_icu_hour": snapshot,
        "total_icu_hours": patient.get("total_icu_hours"),
        "alert": alert,
        "scores": scores,
        "confidence": confidence_for(patient),
        "reasons": reasons_for(patient_id, snapshot),
        "summary": summary_for(patient_id, snapshot),
        "outcome": {"developed_sepsis": patient.get("developed_sepsis"), "note": OUTCOME_NOTE},
    }
