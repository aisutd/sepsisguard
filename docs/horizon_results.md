All three models use the same 36 features and the same tuned settings from Week 3, so only the horizon changes. Early-detection columns are measured against the true onset (from `onsets.csv`) at a 10% false-alarm rate, on 586 sepsis and 7,481 non-sepsis test patients.

| Horizon | AUROC | AUPRC (random) | Warned before onset | ≥6 h early | ≥12 h early | ≥24 h early | Median lead |
|---|---|---|---|---|---|---|---|
| 6 h | 0.808 | 0.095 (0.018) | 49.0% | 39.6% | 35.2% | 29.5% | 37 h |
| 12 h | 0.809 | 0.139 (0.027) | 46.6% | 38.4% | 34.5% | 28.7% | 40 h |
| 24 h | 0.815 | 0.221 (0.042) | 44.2% | 38.2% | 34.5% | 27.8% | 40 h |

As the horizon grew from 6 to 24 hours, AUPRC rose from 0.095 to 0.221, but mostly because more hours are labeled positive (random AUPRC rose from 0.018 to 0.042), so every model stays about 5× better than random, and AUROC stayed near 0.81. The surprise was that, compared at the same 10% false-alarm rate, training for a longer horizon did not make the model warn any earlier: all three warned about 38–40% of sepsis patients at least 6 hours ahead and 28–30% at least 24 hours ahead, differences of only a few patients. At fixed thresholds the 24 h model looks far better (76% warned at 0.3), but only because it also alarms on 35% of non-sepsis patients, which is why models should be compared at a fixed false-alarm rate. Together with the long median lead of about 40 hours, this suggests XGBoost mostly recognizes patients who look high-risk early in their stay rather than timing when sepsis will start, likely because each row only sees the current hour's values; the new change-over-time features would be a good way to test this. Also, 40% of sepsis patients reach onset before ICU hour 24, so warnings of 24+ hours are capped at about 60%.
