import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))  # lets us import from src/
from evaluation.evaluate import evaluate, evaluate_early_detection, find_onsets

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
HORIZONS = [6, 12, 24]
SEED = 42

# Columns the model must NOT learn from: identifiers, and every label column.
# (A label column as a feature would leak the answer, e.g. label_24h gives away label_12h.)
ID_COLS = ["patient_id", "site"]
LABEL_COLS = ["SepsisLabel", "label_6h", "label_12h", "label_24h"]

# Best settings from Week 3 Optuna tuning (tune_baseline.py), reused for every horizon
SETTINGS = dict(
    n_estimators=21,
    max_depth=3,
    learning_rate=0.1499,
    min_child_weight=14,
    subsample=0.8398,
    colsample_bytree=0.7020,
    scale_pos_weight=9.564,
)

# 1. Load Maritza's Week 4 files
train = pd.read_csv(DATA_DIR / "train_features.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")
onsets_file = pd.read_csv(DATA_DIR / "onsets.csv")

# Same 36 features as the Week 3 baseline: everything except IDs, labels and the new change columns
features = [c for c in train.columns if c not in ID_COLS + LABEL_COLS and "_change_" not in c]
print(f"Using {len(features)} features (should be 36)")

# 2. Onset hour for each sepsis patient (an empty onset_hour means never septic)
onsets = onsets_file.dropna(subset=["onset_hour"]).set_index("patient_id")["onset_hour"]

# 3. Sanity checks
#    a) label_6h should be exactly the original SepsisLabel
label_mismatches = int((train["label_6h"] != train["SepsisLabel"]).sum())
print(f"Check: label_6h differs from SepsisLabel in {label_mismatches} rows (should be 0)")

#    b) onsets.csv should agree with the onsets evaluate.py works out itself (first 1 + 6)
both = pd.concat([train, test])
derived = find_onsets(both["patient_id"], both["ICULOS"], both["SepsisLabel"])
compare = pd.DataFrame({"derived": derived, "file": onsets})  # lines up by patient_id
disagree = int((compare["derived"] != compare["file"]).sum())  # also counts patients missing from one side
print(f"Check: onsets.csv disagrees with derived onsets for {disagree} patients (should be 0)")

# 4. Train and evaluate one model per horizon
MODEL_DIR.mkdir(exist_ok=True)
results = []
for h in HORIZONS:
    y_train = train[f"label_{h}h"]
    y_test = test[f"label_{h}h"]
    print(f"\n######## Horizon H = {h} h ({y_train.mean():.1%} of training rows labeled 1) ########")

    model = xgb.XGBClassifier(n_jobs=-1, random_state=SEED, **SETTINGS)
    model.fit(train[features], y_train)
    scores = model.predict_proba(test[features])[:, 1]

    # Hour-level scores are measured against THIS horizon's labels...
    evaluate(y_test, scores, name=f"XGBoost H={h}h")
    # ...but early detection is always measured against the true onset, so all horizons are comparable
    early = evaluate_early_detection(test["patient_id"], test["ICULOS"], test["SepsisLabel"],
                                     scores, name=f"XGBoost H={h}h", onsets=onsets)

    model.save_model(MODEL_DIR / f"xgb_h{h}.json")
    print(f"Saved {MODEL_DIR / f'xgb_h{h}.json'}")

    # Keep the numbers for the summary table, at a 10% false-alarm rate so it's a fair comparison
    at_10 = early[early["setting"] == "10% false alarms"].iloc[0]
    results.append({
        "h": h,
        "auroc": roc_auc_score(y_test, scores),
        "auprc": average_precision_score(y_test, scores),
        "random_auprc": y_test.mean(),
        "warned": at_10["warned"],
        "lead_6": at_10["lead_6"],
        "lead_12": at_10["lead_12"],
        "lead_24": at_10["lead_24"],
        "median_lead": at_10["median_lead"],
    })

# 5. Print a summary table in Markdown, ready to paste into docs/horizon_results.md
print("\n\nSummary (early-detection columns are at a 10% false-alarm rate):\n")
print("| Horizon | AUROC | AUPRC (random) | Warned before onset | ≥6 h early | ≥12 h early | ≥24 h early | Median lead |")
print("|---|---|---|---|---|---|---|---|")
for r in results:
    print(f"| {r['h']} h | {r['auroc']:.3f} | {r['auprc']:.3f} ({r['random_auprc']:.3f}) | {r['warned']:.1%} "
          f"| {r['lead_6']:.1%} | {r['lead_12']:.1%} | {r['lead_24']:.1%} | {r['median_lead']:.0f} h |")
