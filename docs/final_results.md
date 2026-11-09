# Final Results: XGBoost vs TFT

Both models are scored on the same 127,064 test patient-hours that the TFT could score (5,972 patients, 357 with sepsis). Early detection is measured against each patient's true sepsis onset. "Warned 6h+" means the model's first alert came at least 6 hours before onset.

## Slide table (10% false-alarm rate)

A 10% false-alarm rate means each model's cutoff is set so that 10% of non-sepsis patients ever get an alert.

| Model | Horizon | Warned 6h+ | Warned 12h+ | Warned 24h+ | Median warning | AUROC | AUPRC (share of positive hours) |
|---|---|---|---|---|---|---|---|
| XGBoost | 6h | 199 of 357 (55.7%) | 182 (51.0%) | 152 (42.6%) | 41 h | 0.842 | 0.128 (2.5%) |
| TFT | 6h | 186 of 357 (52.1%) | 177 (49.6%) | 149 (41.7%) | 43 h | 0.817 | 0.123 (2.5%) |
| XGBoost | 12h | 198 of 357 (55.5%) | 181 (50.7%) | 148 (41.5%) | 41 h | 0.841 | 0.184 (3.9%) |
| TFT | 12h | 185 of 357 (51.8%) | 175 (49.0%) | 150 (42.0%) | 43 h | 0.819 | 0.184 (3.9%) |
| XGBoost | 24h | 193 of 357 (54.1%) | 179 (50.1%) | 146 (40.9%) | 42 h | 0.850 | 0.291 (6.2%) |
| TFT | 24h | 193 of 357 (54.1%) | 179 (50.1%) | 150 (42.0%) | 43 h | 0.834 | 0.269 (6.2%) |

**Takeaway:** the two models are comparable. XGBoost is slightly ahead at 6h and 12h (about 13 more patients warned 6h+), and they tie at 24h.

## At a 20% false-alarm rate

| Model | Horizon | Warned 6h+ | Warned 12h+ | Warned 24h+ | Median warning |
|---|---|---|---|---|---|
| XGBoost | 6h | 223 (62.5%) | 201 (56.3%) | 164 (45.9%) | 42 h |
| TFT | 6h | 202 (56.6%) | 187 (52.4%) | 159 (44.5%) | 45 h |
| XGBoost | 12h | 220 (61.6%) | 198 (55.5%) | 166 (46.5%) | 42 h |
| TFT | 12h | 207 (58.0%) | 186 (52.1%) | 161 (45.1%) | 43 h |
| XGBoost | 24h | 220 (61.6%) | 201 (56.3%) | 165 (46.2%) | 42 h |
| TFT | 24h | 222 (62.2%) | 204 (57.1%) | 164 (45.9%) | 42 h |

## By site (warned 6h+ at 10% false alarms)

| Model | Horizon | Site A (232 sepsis) | Site B (125 sepsis) |
|---|---|---|---|
| XGBoost | 6h | 134 (57.8%) | 66 (52.8%) |
| TFT | 6h | 126 (54.3%) | 62 (49.6%) |
| XGBoost | 12h | 134 (57.8%) | 65 (52.0%) |
| TFT | 12h | 124 (53.4%) | 60 (48.0%) |
| XGBoost | 24h | 128 (55.2%) | 64 (51.2%) |
| TFT | 24h | 130 (56.0%) | 65 (52.0%) |

The pattern holds at both hospitals. Site B has fewer sepsis patients, so a difference of a few patients moves its percentages more.

## TFT alert cutoffs (used by the app)

Each cutoff is the TFT score at which 10% of non-sepsis patients ever trigger an alert. They were set on Arnav's **validation** scores, then checked on the test set. They were never adjusted after looking at the demo patients.

| Alert level | TFT horizon | Cutoff | Non-sepsis patients alerted (test) |
|---|---|---|---|
| Urgent | 6h | 0.0269 | 10.2% |
| High risk | 12h | 0.0431 | 9.7% |
| Elevated risk | 24h | 0.0646 | 9.1% |

With all three alert levels combined, **14.9%** of non-sepsis patients trigger at least one alert.

## Caveats

- These rows leave out each patient's first 24 hours and patients with too little history for the TFT, so the numbers are higher than the full test set. Compare them only with each other.
- The median warning of about 40 hours mostly reflects patients who look high-risk early in their stay, not precise timing of sepsis onset. The "warned 6h+" share is the more reliable headline.

## Reproduce

python src/evaluation/final_evaluation.py

This prints every table, including the full per-site tables, and rewrites `data/sample/alert_thresholds.json`. Re-run it if the TFT models are updated.
