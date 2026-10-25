import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLDS = [0.3, 0.5, 0.7]      # an alert "fires" when the score is at or above this
FALSE_ALARM_RATES = [0.10, 0.20]  # also compare models at these fixed false-alarm rates
LEAD_TIMES = [6, 12, 24]          # "warned at least this many hours before onset"
LABEL_LEAD_HOURS = 6              # PhysioNet labels hours from 6 h BEFORE onset as 1


def evaluate(y_true, scores, name="model"):
    """Print AUROC, AUPRC and an alert table for one model's predictions."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n_septic = int(y_true.sum())

    print(f"\n=== {name} ===")
    print(f"Rows: {len(y_true):,}   septic rows: {n_septic:,} ({y_true.mean():.1%})")

    # AUROC: how well the model ranks septic rows above non-septic ones (0.5 = random)
    print(f"AUROC: {roc_auc_score(y_true, scores):.3f}")

    # AUPRC: how well it finds the rare septic rows. Random scores equal the
    # share of septic rows, so this is the honest number for imbalanced data
    print(f"AUPRC: {average_precision_score(y_true, scores):.3f}  (random = {y_true.mean():.3f})")

    # Alert table: what would happen if we raised an alarm at each threshold
    print(f"\n{'Threshold':>9} {'Alerts':>10} {'Caught':>10} {'Share caught':>13}")
    for t in THRESHOLDS:
        alerts = scores >= t                            # True where an alert fires
        caught = int((alerts & (y_true == 1)).sum())    # alerts on rows that really are septic
        share = caught / n_septic if n_septic else 0.0
        print(f"{t:>9} {int(alerts.sum()):>10,} {caught:>10,} {share:>13.1%}")


def find_onsets(patient_ids, hours, y_true, label_lead_hours=LABEL_LEAD_HOURS):
    
    df = pd.DataFrame({"patient_id": np.asarray(patient_ids),
                       "hour": np.asarray(hours),
                       "label": np.asarray(y_true)})
    first_positive_hour = df[df["label"] == 1].groupby("patient_id")["hour"].min()
    return first_positive_hour + label_lead_hours


def _early_detection_at(df, onsets, threshold):
    is_septic_row = df["patient_id"].isin(onsets.index)
    alerts = df[df["score"] >= threshold]
    alert_is_septic = alerts["patient_id"].isin(onsets.index)

    # False alarms: share of NON-sepsis patients who get at least one alert, ever
    n_non_septic = df.loc[~is_septic_row, "patient_id"].nunique()
    alerted_non_septic = alerts.loc[~alert_is_septic, "patient_id"].nunique()

    # Early warning: for each sepsis patient, their FIRST alert at or before onset.
    # Alerts after onset don't count as a warning; sepsis has already started.
    septic_alerts = alerts[alert_is_septic]
    septic_alerts = septic_alerts[septic_alerts["hour"] <= septic_alerts["patient_id"].map(onsets)]
    first_alert = septic_alerts.groupby("patient_id")["hour"].min()

    # Lead time = how many hours before onset that first alert came
    lead = onsets.loc[first_alert.index] - first_alert

    n_septic = len(onsets)
    result = {
        "threshold": threshold,
        "false_alarm": alerted_non_septic / n_non_septic,
        "warned": len(lead) / n_septic,                    # share warned at all before onset
        "median_lead": lead.median() if len(lead) else float("nan"),  # among warned patients
    }
    for h in LEAD_TIMES:
        # Share of ALL sepsis patients warned at least h hours early
        result[f"lead_{h}"] = (lead >= h).sum() / n_septic
    return result


def _threshold_for_false_alarm_rate(df, onsets, rate):
    """Find the threshold at which `rate` (e.g. 10%) of non-sepsis patients get an alert.

    A patient alerts if their HIGHEST score reaches the threshold, so we take the
    (1 - rate) quantile of non-sepsis patients' highest scores.
    """
    non_septic = df[~df["patient_id"].isin(onsets.index)]
    highest_score = non_septic.groupby("patient_id")["score"].max()
    return float(highest_score.quantile(1 - rate))


def evaluate_early_detection(patient_ids, hours, y_true, scores, name="model", onsets=None):
    """Print patient-level early-detection numbers at fixed thresholds and at
    fixed false-alarm rates. Pass `onsets` (patient_id -> onset hour) to use known
    onsets; otherwise they're worked out from the labels with find_onsets()."""
    df = pd.DataFrame({"patient_id": np.asarray(patient_ids),
                       "hour": np.asarray(hours),
                       "score": np.asarray(scores)})
    if onsets is None:
        onsets = find_onsets(patient_ids, hours, y_true)
    onsets = onsets[onsets.index.isin(df["patient_id"].unique())]  # only patients in this data

    # Fixed thresholds
    rows = []
    for t in THRESHOLDS:
        rows.append({"setting": f"threshold {t}", **_early_detection_at(df, onsets, t)})

    # Fixed false-alarm rates: the fair way to compare two models, since each
    # model gets whatever threshold gives it the same number of false alarms
    for rate in FALSE_ALARM_RATES:
        t = _threshold_for_false_alarm_rate(df, onsets, rate)
        rows.append({"setting": f"{rate:.0%} false alarms", **_early_detection_at(df, onsets, t)})

    n_non_septic = df["patient_id"].nunique() - len(onsets)
    print(f"\n--- {name}: early detection (per patient) ---")
    print(f"Sepsis patients: {len(onsets):,}   non-sepsis patients: {n_non_septic:,}")
    print(f"{'Setting':<18}{'Thresh':>8}{'False alarms':>14}{'Warned':>9}{'Median lead':>13}"
          + "".join(f"{f'>={h}h':>8}" for h in LEAD_TIMES))
    for r in rows:
        print(f"{r['setting']:<18}{r['threshold']:>8.3f}{r['false_alarm']:>14.1%}{r['warned']:>9.1%}"
              f"{r['median_lead']:>12.1f}h"
              + "".join(f"{r[f'lead_{h}']:>8.1%}" for h in LEAD_TIMES))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import xgboost as xgb

    test = pd.read_csv("data/processed/test.csv")
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")
    features = model.get_booster().feature_names  # the 36 columns the model was trained on
    scores = model.predict_proba(test[features])[:, 1]

    evaluate(test["SepsisLabel"], scores, name="XGBoost baseline")
    evaluate_early_detection(test["patient_id"], test["ICULOS"], test["SepsisLabel"], scores,
                             name="XGBoost baseline")
