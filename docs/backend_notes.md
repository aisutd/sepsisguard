# Backend notes

The backend is a FastAPI server in `backend/main.py`. It only reads the shared files in `data/sample/` (see `docs/demo_data_contract.md`) and sends them to the dashboard.

## Start it (one command, from `sepsisguard/`)

uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000/docs to try every endpoint in the browser. CORS allows the React dashboard at `http://localhost:5173`.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` if the server is running |
| `GET /patients` | The 8 demo patients, each with site, snapshot hour, alert level, confidence label and outcome |
| `GET /patients/{patient_id}` | One patient at their snapshot hour: alert level, TFT scores at the snapshot, score history up to the snapshot (each hour with its alert level), confidence, reasons (SHAP), summary and outcome |
| `POST /confidence` | Takes one raw patient row and returns its confidence label and missing fields |

## Alert level

Uses the TFT cutoffs in `data/sample/alert_thresholds.json` (see `docs/final_results.md`), checked in order:
6h score at or above its cutoff = **Urgent**, else 12h = **High risk**, else 24h = **Elevated risk**, else **No alert**.

## Confidence

From `imputed_share` in `demo_patients.json` (the share of the 35 inputs filled in rather than measured), using `label_from_share` in `backend/confidence.py`:
under 10% = **high**, over 40% = **low**, otherwise **medium**.

## Missing data

- A missing file or unknown patient returns a clear "not available" message (503 or 404) instead of crashing.
- Reasons and summaries are read from any `data/sample/` file with "shap" or "summar" in its name, but only entries for that patient at their `snapshot_icu_hour` (field `"icu_hour"`) are used, as the demo contract requires. Otherwise the section says "not available yet".

## Notes

- Scores are relative risk for ranking patients, not probabilities, so they are never shown as percentages.
- The outcome is labelled "demo only", because a real ICU would not know it yet.
