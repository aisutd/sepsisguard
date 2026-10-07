import math

VITALS= ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

LABS= ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
       "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Glucose",
       "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
       "Hct", "Hgb", "PTT", "WBC", "Platelets"]

DEMOGRAPHICS= ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

EXPECTED_FIELDS= VITALS + LABS + DEMOGRAPHICS

HIGH_MAX_MISSING= 0.10
LOW_MIN_MISSING= 0.40

def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False

def score_confidence(row):
    missing_fields= []
    for field in EXPECTED_FIELDS:
        if is_missing(row.get(field)):
            missing_fields.append(field)

    total= len(EXPECTED_FIELDS)
    missing_count= len(missing_fields)
    missing_fraction= missing_count/total

    if missing_fraction < HIGH_MAX_MISSING:
        label= "high"
    elif missing_fraction > LOW_MIN_MISSING:
        label= "low"
    else:
        label= "medium"

    return{
        "confidence": label,
        "missing_count": missing_count,
        "total_fields": total,
        "missing_fraction": round(missing_fraction, 3),
        "missing_fields": missing_fields
        }

