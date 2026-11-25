// featureNames.js
// Plain names for the model's input columns, for display only.
//
// The data (feature_name in the SHAP and summary files) keeps the original
// column codes, and so do the summary generator and checker
// (src/nlp_pipeline/feature_names.py). Only the label on screen changes, so
// the checker keeps working. To rename something on screen, change it here.
const PLAIN_NAMES = {
  // Vital signs
  HR: "Heart rate",
  O2Sat: "Blood oxygen level",
  Temp: "Temperature",
  SBP: "Blood pressure (top number)",
  MAP: "Average blood pressure",
  DBP: "Blood pressure (bottom number)",
  Resp: "Breathing rate",
  // Lab results
  BaseExcess: "Base excess (blood acid balance)",
  HCO3: "Bicarbonate",
  FiO2: "Oxygen support",
  pH: "Blood acidity (pH)",
  PaCO2: "Carbon dioxide in the blood",
  SaO2: "Oxygen in arterial blood",
  AST: "Liver enzyme (AST)",
  BUN: "Kidney waste level (BUN)",
  Alkalinephos: "Liver enzyme (alkaline phosphatase)",
  Calcium: "Calcium",
  Chloride: "Chloride",
  Creatinine: "Creatinine (kidney function)",
  Glucose: "Blood sugar",
  Lactate: "Lactate",
  Magnesium: "Magnesium",
  Phosphate: "Phosphate",
  Potassium: "Potassium",
  Bilirubin_total: "Bilirubin (liver)",
  Hct: "Red blood cell share (hematocrit)",
  Hgb: "Hemoglobin",
  PTT: "Clotting time (PTT)",
  WBC: "White blood cells",
  Platelets: "Platelets",
  // Patient and timing
  Age: "Age",
  Gender: "Sex",
  Unit1: "Medical ICU",
  Unit2: "Surgical ICU",
  HospAdmTime: "Time from hospital to ICU admission",
  ICULOS: "Hours in the ICU",
};

// The plain name for a column code. Unknown codes are shown as they are,
// so a new feature never disappears from the screen.
export function featureLabel(code) {
  return PLAIN_NAMES[code] ?? code;
}
