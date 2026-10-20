# Backend notes
The backend is a FastAPI server in 'backend/main.py' with three endpoints.
'GET  /health' returns '{"status": "ok"}' and is used to check that the server is running.
'GET /patients' returns the 20 sample patients from 'data/sample/patients_sample.json', each with their latest heart rate, respiratory rate and temperature, whether they developed sepsis, and a data-confidence label
'POST /confidence' takes one patient row as JSON and returns its confidence label (high/medium/low), how many fields are missing, and which ones, using 'score_confidence' from 'backend/confidence.py'