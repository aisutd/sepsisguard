# Demo data notes

Build command (from `sepsisguard/`): `python src/demo/build_demo_files.py`
Outputs: `data/sample/demo_patients.json`, `data/sample/demo_scores.json`.
Locked choices: `data/sample/demo_selection.json` (do not delete; everyone uses these IDs).

## How the 8 demo patients were chosen
The rule was fixed before looking at any model output.
1. Start from every patient in `data/processed/test.csv`.
2. Keep patients the TFT can score: 25+ hourly rows (24 hours of history plus 1 hour to predict). That leaves 5,972 patients, 357 of whom developed sepsis. This matches the 5,972 patients Arnav's TFT scored.
3. Sort each group (developed sepsis / did not) by `patient_id` and draw 4 from each with `pandas.DataFrame.sample(random_state=42)`.

| Developed sepsis | Did not |
|---|---|
| p003455 (A) | p000619 (A) |
| p019680 (A) | p105904 (B) |
| p109388 (B) | p109064 (B) |
| p111054 (B) | p111162 (B) |

## Snapshot hour
`snapshot_icu_hour` is the ICU hour with the patient's highest 12h TFT score (the earliest hour if there is a tie). It is chosen once, saved in `demo_selection.json`, and never recomputed, even if the models are retrained.
The hours were chosen with the first TFT scores. On 2026-11-11, `demo_scores.json` was rebuilt with Arnav's final 12h model (the 6h and 24h files were unchanged). The locked hours were kept, so for 4 of the 8 patients the snapshot hour is no longer their highest final 12h score (e.g. p003455: snapshot 113, final peak 94).

## Imputed share
For the snapshot hour I look up the patient's raw `.psv` file. Each of the 35 model inputs that is blank there counts as filled in: in `test.csv` it was carried forward from an earlier hour or set to the training median. `imputed_share` = filled-in inputs / 35. If the hour is missing from the raw file entirely, the cleaning step added it, so the share is 1.0. This is the same count that `backend/confidence.py` makes, so the `confidence` label in the file matches the `/confidence` endpoint.
