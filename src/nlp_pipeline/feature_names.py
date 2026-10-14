"""Feature vocabulary shared by the summary generator and the verifier.

Every feature the baseline model can use (the PhysioNet 2019 columns) is listed
here with:
  - a readable name the generator writes into summaries, and
  - aliases: other ways the same feature might be written in free text.

The verifier uses the aliases to spot ANY feature a summary mentions, so it can
catch a summary that names something outside the patient's top 5, even when
the wording differs from ours (for example "pulse" instead of "heart rate").
"""

# Column name -> (readable name, other ways it might appear in text)
FEATURES = {
    # Vital signs
    "HR": ("heart rate", ["pulse", "HR"]),
    "O2Sat": ("oxygen saturation", ["SpO2", "O2 sat", "O2Sat", "oxygen sat"]),
    "Temp": ("temperature", ["body temperature", "fever"]),
    "SBP": ("systolic blood pressure", ["systolic", "SBP"]),
    "MAP": ("mean arterial pressure", ["MAP"]),
    "DBP": ("diastolic blood pressure", ["diastolic", "DBP"]),
    "Resp": ("respiratory rate", ["breathing rate", "respiration"]),
    "EtCO2": ("end-tidal CO2", ["EtCO2", "end tidal CO2"]),
    # Lab results
    "BaseExcess": ("base excess", []),
    "HCO3": ("bicarbonate", ["HCO3"]),
    "FiO2": ("fraction of inspired oxygen", ["FiO2", "inspired oxygen"]),
    "pH": ("blood pH", ["pH"]),
    "PaCO2": ("arterial CO2 pressure", ["PaCO2"]),
    "SaO2": ("arterial oxygen saturation", ["SaO2"]),
    "AST": ("AST", ["aspartate aminotransferase"]),
    "BUN": ("blood urea nitrogen", ["BUN", "urea"]),
    "Alkalinephos": ("alkaline phosphatase", ["ALP"]),
    "Calcium": ("calcium", []),
    "Chloride": ("chloride", []),
    "Creatinine": ("creatinine", []),
    "Bilirubin_direct": ("direct bilirubin", []),
    "Glucose": ("glucose", ["blood sugar"]),
    "Lactate": ("lactate", ["lactic acid"]),
    "Magnesium": ("magnesium", []),
    "Phosphate": ("phosphate", []),
    "Potassium": ("potassium", []),
    "Bilirubin_total": ("total bilirubin", []),
    "TroponinI": ("troponin I", ["troponin"]),
    "Hct": ("hematocrit", ["Hct"]),
    "Hgb": ("hemoglobin", ["Hgb"]),
    "PTT": ("partial thromboplastin time", ["PTT"]),
    "WBC": ("white blood cell count", ["WBC", "white cell count", "leukocytes"]),
    "Fibrinogen": ("fibrinogen", []),
    "Platelets": ("platelet count", ["platelets"]),
    # Demographics and timing
    "Age": ("age", []),
    "Gender": ("sex", ["gender"]),
    "Unit1": ("being in a medical ICU", ["MICU", "medical ICU"]),
    "Unit2": ("being in a surgical ICU", ["SICU", "surgical ICU"]),
    "HospAdmTime": ("time since hospital admission", ["hospital admission"]),
    "ICULOS": ("time in the ICU", ["ICU length of stay", "ICULOS"]),
}

# Generic phrases that point to a group of features rather than one.
# "blood pressure" alone could mean SBP, MAP or DBP, so it counts as
# grounded if ANY of those is in the patient's top 5.
GROUP_TERMS = {
    "blood pressure": {"SBP", "MAP", "DBP"},
    "bilirubin": {"Bilirubin_direct", "Bilirubin_total"},
    "oxygen": {"O2Sat", "SaO2", "FiO2"},
    "CO2": {"EtCO2", "PaCO2"},
    "ICU type": {"Unit1", "Unit2"},
    "ICU unit": {"Unit1", "Unit2"},
}


def readable_name(feature):
    """Return the name we print for a column, e.g. 'HR' -> 'heart rate'."""
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature}")
    return FEATURES[feature][0]
