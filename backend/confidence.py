"""
confidence.py

Says how much to trust a patient's data: high, medium or low confidence,
based on how much of it was missing (or filled in) instead of measured.

Rule (also written in docs/backend_notes.md):
    under 10% missing or filled in  -> high
    over 40%                        -> low
    anything in between             -> medium
"""

import math

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Glucose",
        "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
        "Hct", "Hgb", "PTT", "WBC", "Platelets"]

DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

EXPECTED_FIELDS = VITALS + LABS + DEMOGRAPHICS

HIGH_MAX_MISSING = 0.10   # below this share missing -> "high"
LOW_MIN_MISSING = 0.40    # above this share missing -> "low"


def is_missing(value):
    """True if a field is empty: None, NaN or a blank string."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def label_from_share(missing_share):
    """Turn a share of missing or filled-in inputs (0 to 1) into a confidence label.
    Used by score_confidence() below, and by the backend with the imputed_share
    from demo_patients.json, so both follow the same rule."""
    if missing_share < HIGH_MAX_MISSING:
        return "high"
    if missing_share > LOW_MIN_MISSING:
        return "low"
    return "medium"


def score_confidence(row):
    """Check one patient row for missing fields and return its confidence label."""
    missing_fields = [field for field in EXPECTED_FIELDS if is_missing(row.get(field))]

    total = len(EXPECTED_FIELDS)
    missing_count = len(missing_fields)
    missing_fraction = missing_count / total

    return {
        "confidence": label_from_share(missing_fraction),
        "missing_count": missing_count,
        "total_fields": total,
        "missing_fraction": round(missing_fraction, 3),
        "missing_fields": missing_fields,
    }
