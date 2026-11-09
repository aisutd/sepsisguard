import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))  # lets us import from src/
from evaluation.evaluate import _early_detection_at, _threshold_for_false_alarm_rate

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
THRESHOLDS_PATH = Path("data/sample/alert_thresholds.json")
KEYS = ["site", "patient_id", "ICULOS"]   # identifies one patient-hour
HORIZONS = [6, 12, 24]
FALSE_ALARM_RATES = [0.10, 0.20]          # the evidence table is reported at both
CUTOFF_RATE = 0.10                        # the alert cutoffs use 10%
ALERT_LEVELS = {6: "Urgent", 12: "High risk", 24: "Elevated risk"}


test = pd.read_csv(DATA_DIR / "test_features.csv")

# True onset hour for each sepsis patient (patients not listed never had sepsis)
onsets_file = pd.read_csv(DATA_DIR / "onsets.csv")
onsets = onsets_file.dropna(subset=["onset_hour"]).set_index("patient_id")["onset_hour"]

# TFT test scores. A missing horizon is skipped, so the script still runs while
# we wait for a file. Just re-run it once the file arrives.
tft = {}
for h in HORIZONS:
    path = DATA_DIR / f"tft_scores_{h}h.csv"
    if path.exists():
        tft[h] = pd.read_csv(path)
    else:
        print(f"NOTE: {path.name} not found, so the {h}h TFT is skipped for now")
horizons = sorted(tft)



# Score every test hour with each XGBoost horizon model
for h in horizons:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / f"xgb_h{h}.json")
    features = model.get_booster().feature_names  # the 36 columns it was trained on
    test[f"xgb_{h}h"] = model.predict_proba(test[features])[:, 1]

rows = None
for h in horizons:
    part = tft[h][KEYS + ["risk_score"]].rename(columns={"risk_score": f"tft_{h}h"})
    rows = part if rows is None else rows.merge(part, on=KEYS, how="inner")

# Attach the true labels and XGBoost's scores to those rows
extra = [f"label_{h}h" for h in horizons] + [f"xgb_{h}h" for h in horizons]
rows = rows.merge(test[KEYS + extra], on=KEYS, how="left")
unmatched = int(rows[extra].isna().any(axis=1).sum())
print(f"Same-row set: {len(rows):,} patient-hours, {rows['patient_id'].nunique():,} patients")
print(f"Check: rows with no match in test_features.csv: {unmatched} (should be 0)")
rows = rows.dropna(subset=extra)


def patient_level(df, score_col, rate):
    """Patient-level early-detection numbers for one model at a fixed false-alarm rate."""
    # evaluate.py's helpers expect columns named patient_id, hour and score
    data = df[["patient_id", "ICULOS", score_col]].rename(
        columns={"ICULOS": "hour", score_col: "score"})
    in_data = onsets[onsets.index.isin(data["patient_id"].unique())]
    threshold = _threshold_for_false_alarm_rate(data, in_data, rate)
    result = _early_detection_at(data, in_data, threshold)
    result["n_septic"] = len(in_data)
    # The shares are counts divided by the number of sepsis patients, so turn them back into counts
    for lead in (6, 12, 24):
        result[f"count_{lead}"] = round(result[f"lead_{lead}"] * len(in_data))
    return result


def evidence_table(df, title):
    """Print one Markdown table: both models, every horizon, both false-alarm rates.
    The false-alarm rate is set within this group of patients (e.g. within Site A)."""
    n_septic = int(onsets.index.isin(df["patient_id"].unique()).sum())
    print(f"\n### {title}: {df['patient_id'].nunique():,} patients, {n_septic} with sepsis\n")
    print("| Model | Horizon | False alarms | Median warning | Warned 6h+ | Warned 12h+ "
          "| Warned 24h+ | AUROC | AUPRC (share of positive hours) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for h in horizons:
        y = df[f"label_{h}h"]
        for name, col in [("XGBoost", f"xgb_{h}h"), ("TFT", f"tft_{h}h")]:
            auroc = roc_auc_score(y, df[col])     # hour-level ranking quality
            auprc = average_precision_score(y, df[col])
            for rate in FALSE_ALARM_RATES:
                r = patient_level(df, col, rate)
                n = r["n_septic"]
                print(f"| {name} | {h}h | {rate:.0%} | {r['median_lead']:.0f} h "
                      f"| {r['lead_6']:.1%} ({r['count_6']} of {n}) "
                      f"| {r['lead_12']:.1%} ({r['count_12']} of {n}) "
                      f"| {r['lead_24']:.1%} ({r['count_24']} of {n}) "
                      f"| {auroc:.3f} | {auprc:.3f} ({y.mean():.1%}) |")


evidence_table(rows, "All patients")
for site in sorted(rows["site"].unique()):
    evidence_table(rows[rows["site"] == site], f"Site {site}")


cutoffs, sources = {}, {}
for h in horizons:
    # Prefer validation scores, so the cutoffs aren't set on the same data we report on
    val_path = DATA_DIR / f"tft_val_scores_{h}h.csv"
    if val_path.exists():
        scores, sources[h] = pd.read_csv(val_path), "validation"
    else:
        scores, sources[h] = tft[h], "test"
    data = scores.rename(columns={"risk_score": "score"})[["patient_id", "score"]]
    in_data = onsets[onsets.index.isin(data["patient_id"].unique())]
    cutoffs[h] = _threshold_for_false_alarm_rate(data, in_data, CUTOFF_RATE)


non_septic = rows[~rows["patient_id"].isin(onsets.index)]
n_non_septic = non_septic["patient_id"].nunique()
any_alert = pd.Series(False, index=non_septic.index)
each_rate = {}
for h in horizons:
    fires = non_septic[f"tft_{h}h"] >= cutoffs[h]           # hours where this cutoff alerts
    each_rate[h] = non_septic.loc[fires, "patient_id"].nunique() / n_non_septic
    any_alert |= fires                                      # alert if ANY horizon fires
combined = non_septic.loc[any_alert, "patient_id"].nunique() / n_non_septic

print("\n### TFT alert cutoffs\n")
print("| Horizon | Alert level | Cutoff | Set from | Non-sepsis patients alerted (test) |")
print("|---|---|---|---|---|")
for h in horizons:
    print(f"| {h}h | {ALERT_LEVELS[h]} | {cutoffs[h]:.4f} | {sources[h]} scores | {each_rate[h]:.1%} |")
print(f"\nAll cutoffs combined: {combined:.1%} of non-sepsis patients trigger at least one alert")


output = {
    "model": "TFT",
    "created": date.today().isoformat(),
    "rule": "Each cutoff is the score at which 10% of non-sepsis patients ever trigger an alert",
    "alert_order": "6h at or above its cutoff = Urgent, else 12h = High risk, "
                   "else 24h = Elevated risk, else No alert",
    "thresholds": {f"{h}h": round(cutoffs[h], 6) for h in horizons},
    "alert_levels": {f"{h}h": ALERT_LEVELS[h] for h in horizons},
    "scores_used_for_cutoffs": {f"{h}h": sources[h] for h in horizons},
    "false_alarm_rate_on_test": {f"{h}h": round(each_rate[h], 4) for h in horizons},
    "combined_false_alarm_rate_on_test": round(combined, 4),
}
THRESHOLDS_PATH.write_text(json.dumps(output, indent=2))
print(f"\nSaved {THRESHOLDS_PATH}")
