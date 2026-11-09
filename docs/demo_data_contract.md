# Demo data contract

These are the shared files for the final demo screen. Every component reads its data from them.
They are built by `python src/demo/build_demo_files.py` (run it from `sepsisguard/`). Don't edit them by hand.
The selection rule and the reasoning behind it are in `docs/demo_data_notes.md`.

## The 8 demo patients (locked)

| patient_id | site | developed sepsis | snapshot_icu_hour |
|---|---|---|---|
| p003455 | A | yes | 113 |
| p019680 | A | yes | 70 |
| p109388 | B | yes | 36 |
| p111054 | B | yes | 35 |
| p000619 | A | no | 28 |
| p105904 | B | no | 36 |
| p109064 | B | no | 26 |
| p111162 | B | no | 44 |

These IDs and hours are fixed in `data/sample/demo_selection.json` and will not change, even if the models are retrained.
**Anything you show for a patient should describe that patient at their `snapshot_icu_hour`.**

## `data/sample/demo_patients.json`

A list of 8 objects, one per patient, in the order shown in the table above.

```json
{
  "patient_id": "p003455",
  "site": "A",
  "snapshot_icu_hour": 113,
  "total_icu_hours": 113,
  "developed_sepsis": true,
  "imputed_share": 0.4,
  "imputed_count": 14,
  "total_inputs": 35,
  "imputed_fields": ["Lactate", "..."],
  "confidence": "medium",
  "inputs_at_snapshot": { "HR": 98.0, "Temp": 37.2, "...": 0.0 }
}
```

| Field | Type | Meaning |
|---|---|---|
| `patient_id` | string | Patient ID, e.g. `p003455`. This is the key used to join with every other file. |
| `site` | `"A"` or `"B"` | Hospital the patient came from. |
| `snapshot_icu_hour` | int | The ICU hour the screen shows: the hour of this patient's highest 12h TFT score. |
| `total_icu_hours` | int | How many hours the patient's ICU stay lasted. |
| `developed_sepsis` | bool | The real outcome. On screen, always label it **"demo only"**, because a real ICU would not know it yet. |
| `imputed_share` | float, 0–1 | Share of the 35 model inputs that were filled in at the snapshot hour instead of measured. |
| `imputed_count` / `total_inputs` | int | The same share as counts, e.g. 14 of 35. |
| `imputed_fields` | list of strings | Which inputs were filled in. |
| `confidence` | `"high"`, `"medium"` or `"low"` | The same rule as the backend's `/confidence` endpoint: high under 10% filled in, low over 40%, medium in between. |
| `inputs_at_snapshot` | object | The cleaned values the model saw for all 35 inputs at the snapshot hour, rounded to 2 decimals. |

## `data/sample/demo_scores.json`

A list of 8 objects, one per patient, in the same order.

```json
{
  "patient_id": "p003455",
  "snapshot_icu_hour": 113,
  "scores": { "6h": 0.0971, "12h": 0.5815, "24h": 0.081 },
  "history": [
    { "icu_hour": 25, "6h": 0.0048, "12h": 0.0041, "24h": 0.0057 },
    "...one entry per hour up to and including the snapshot hour"
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `scores` | object | TFT risk scores at the snapshot hour for the 6h, 12h and 24h horizons, rounded to 4 decimals. |
| `history` | list | One entry per hour from ICU hour 25 (the first hour the TFT can score) up to the snapshot hour, for trend lines. Hours after the snapshot are left out, so the screen never shows the future. |

**These scores are relative risk for ranking patients. They are not probabilities.** Don't display them as percentages.

## Adding your own data (SHAP, summaries, etc.)

If your component brings its own file:
- Include one entry for each of the 8 `patient_id`s above.
- Compute it at that patient's `snapshot_icu_hour` and store the hour in the entry as `"icu_hour"`.
- Name the horizon if it applies, e.g. `"horizon": 12`.

Example (SHAP top features):
```json
{
  "patient_id": "p003455",
  "icu_hour": 113,
  "horizon": 12,
  "top_features": [ { "feature_name": "Lactate", "shap_value": 0.0029 } ]
}
```
