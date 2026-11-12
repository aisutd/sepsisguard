# Final Results: XGBoost vs TFT

Both models are scored on the same 127,064 test patient-hours that the TFT could score (5,972 patients, 357 with sepsis). Early detection is measured against each patient's true sepsis onset. "Warned 6h+" means the model's first alert came at least 6 hours before onset.

## How the alert cutoffs were set

- **Data:** Arnav's final TFT **validation** scores, i.e. patients not in the test set.
- **Rule:** each cutoff is the TFT score at which **10% of non-sepsis patients ever trigger an alert** (the 90th percentile of each non-sepsis patient's highest score).
- **Checked on test:** the cutoffs alert on 10.2%, 9.0% and 9.1% of non-sepsis test patients. All three combined: **14.5%**.
- The cutoffs were never adjusted after looking at the demo patients. They were re-run once, unchanged in method, when Arnav locked his final models (only the 12h model changed).

| Alert level | TFT horizon | Cutoff | Non-sepsis patients alerted (test) |
|---|---|---|---|
| Urgent | 6h | 0.0269 | 10.2% |
| High risk | 12h | 0.0343 | 9.0% |
| Elevated risk | 24h | 0.0646 | 9.1% |

The app checks them in order: 6h at or above its cutoff = Urgent, else 12h = High risk, else 24h = Elevated risk, else No alert.

## Evidence table (10% false-alarm rate)

| Model | Horizon | Warned 6h+ | Warned 12h+ | Warned 24h+ | Median warning | AUROC | AUPRC (share of positive hours) |
|---|---|---|---|---|---|---|---|
| XGBoost | 6h | 199 of 357 (55.7%) | 182 (51.0%) | 152 (42.6%) | 41 h | 0.842 | 0.128 (2.5%) |
| TFT | 6h | 186 of 357 (52.1%) | 177 (49.6%) | 149 (41.7%) | 43 h | 0.817 | 0.123 (2.5%) |
| XGBoost | 12h | 198 of 357 (55.5%) | 181 (50.7%) | 148 (41.5%) | 41 h | 0.841 | 0.184 (3.9%) |
| TFT | 12h | 191 of 357 (53.5%) | 176 (49.3%) | 146 (40.9%) | 41 h | 0.810 | 0.186 (3.9%) |
| XGBoost | 24h | 193 of 357 (54.1%) | 179 (50.1%) | 146 (40.9%) | 42 h | 0.850 | 0.291 (6.2%) |
| TFT | 24h | 193 of 357 (54.1%) | 179 (50.1%) | 150 (42.0%) | 43 h | 0.834 | 0.269 (6.2%) |

**Takeaway:** the two models are comparable. XGBoost is slightly ahead at 6h (13 more patients warned 6h+) and 12h (7 more), and they tie at 24h.

## Evidence table (20% false-alarm rate)

| Model | Horizon | Warned 6h+ | Warned 12h+ | Warned 24h+ | Median warning |
|---|---|---|---|---|---|
| XGBoost | 6h | 223 (62.5%) | 201 (56.3%) | 164 (45.9%) | 42 h |
| TFT | 6h | 202 (56.6%) | 187 (52.4%) | 159 (44.5%) | 45 h |
| XGBoost | 12h | 220 (61.6%) | 198 (55.5%) | 166 (46.5%) | 42 h |
| TFT | 12h | 206 (57.7%) | 191 (53.5%) | 158 (44.3%) | 41 h |
| XGBoost | 24h | 220 (61.6%) | 201 (56.3%) | 165 (46.2%) | 42 h |
| TFT | 24h | 222 (62.2%) | 204 (57.1%) | 164 (45.9%) | 42 h |

## Per site (10% false-alarm rate, set within each site)

**Site A**: 3,086 patients, 232 with sepsis

| Model | Horizon | Warned 6h+ | Warned 12h+ | Warned 24h+ | AUROC |
|---|---|---|---|---|---|
| XGBoost | 6h | 134 of 232 (57.8%) | 121 (52.2%) | 98 (42.2%) | 0.832 |
| TFT | 6h | 126 of 232 (54.3%) | 118 (50.9%) | 96 (41.4%) | 0.797 |
| XGBoost | 12h | 134 of 232 (57.8%) | 120 (51.7%) | 96 (41.4%) | 0.836 |
| TFT | 12h | 128 of 232 (55.2%) | 116 (50.0%) | 97 (41.8%) | 0.801 |
| XGBoost | 24h | 128 of 232 (55.2%) | 119 (51.3%) | 97 (41.8%) | 0.848 |
| TFT | 24h | 130 of 232 (56.0%) | 122 (52.6%) | 98 (42.2%) | 0.834 |

**Site B**: 2,886 patients, 125 with sepsis

| Model | Horizon | Warned 6h+ | Warned 12h+ | Warned 24h+ | AUROC |
|---|---|---|---|---|---|
| XGBoost | 6h | 66 of 125 (52.8%) | 61 (48.8%) | 55 (44.0%) | 0.844 |
| TFT | 6h | 62 of 125 (49.6%) | 58 (46.4%) | 50 (40.0%) | 0.827 |
| XGBoost | 12h | 65 of 125 (52.0%) | 59 (47.2%) | 51 (40.8%) | 0.836 |
| TFT | 12h | 63 of 125 (50.4%) | 58 (46.4%) | 51 (40.8%) | 0.806 |
| XGBoost | 24h | 64 of 125 (51.2%) | 59 (47.2%) | 49 (39.2%) | 0.841 |
| TFT | 24h | 65 of 125 (52.0%) | 60 (48.0%) | 48 (38.4%) | 0.817 |

The pattern holds at both hospitals. Site B has fewer sepsis patients, so a difference of a few patients moves its percentages more.

## Caveats

- These rows leave out each patient's first 24 hours and patients with too little history for the TFT, so the numbers are higher than the full test set. Compare them only with each other.
- The median warning of about 40 hours mostly reflects patients who look high-risk early in their stay, not precise timing of sepsis onset. The "warned 6h+" share is the more reliable headline.

## Reproduce

python src/evaluation/final_evaluation.py

This prints every table and rewrites `data/sample/alert_thresholds.json`. Re-run it if the TFT models are updated.
