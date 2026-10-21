import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLDS = [0.3, 0.5, 0.7]


def evaluate(y_true, scores, name="model"):
    """Print AUROC, AUPRC and an alert table for one model's predictions."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n_septic = int(y_true.sum())

    print(f"\n=== {name} ===")
    print(f"Rows: {len(y_true):,}   septic rows: {n_septic:,} ({y_true.mean():.1%})")

    print(f"AUROC: {roc_auc_score(y_true, scores):.3f}")

    print(f"AUPRC: {average_precision_score(y_true, scores):.3f}  (random = {y_true.mean():.3f})")

    print(f"\n{'Threshold':>9} {'Alerts':>10} {'Caught':>10} {'Share caught':>13}")
    for t in THRESHOLDS:
        alerts = scores >= t                            
        caught = int((alerts & (y_true == 1)).sum())     
        share = caught / n_septic if n_septic else 0.0
        print(f"{t:>9} {int(alerts.sum()):>10,} {caught:>10,} {share:>13.1%}")


if __name__ == "__main__":
    fake_labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    fake_scores = [0.1, 0.2, 0.1, 0.4, 0.6, 0.2, 0.1, 0.3, 0.8, 0.45]
    evaluate(fake_labels, fake_scores, name="fake example")
