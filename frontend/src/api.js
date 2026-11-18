// api.js
// The one place the frontend gets its data. Components never call fetch()
// themselves; they call the functions below.
//
// All data comes from Harish's FastAPI backend (backend/main.py). It reads the
// shared files in data/sample/ and applies the alert rule and confidence
// labels, so those rules live only in the backend. The frontend only shows
// what the backend returns.
//
// There is no local fallback on purpose: if the backend isn't running, these
// functions throw an error and the screen says how to start it.

// Where the backend runs (see "How to run" in README.md)
const API_URL = "http://localhost:8000";

// Fetch one backend URL and return its JSON.
// Throws an Error with a readable message if something goes wrong.
async function getJson(path) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`);
  } catch {
    // fetch() itself fails when nothing is listening on port 8000
    throw new Error(
      "Backend not running. Start it with: cd backend; uvicorn main:app --reload --port 8000"
    );
  }

  if (!response.ok) {
    // FastAPI puts its error message in "detail",
    // e.g. "Patient p999 not available. Demo patients: ..."
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request to ${path} failed (${response.status})`);
  }

  return response.json();
}

// The 8 demo patients for the bed grid. Each has patient_id, site,
// snapshot_icu_hour, alert_level, confidence and outcome.
export function getPatients() {
  return getJson("/patients");
}

// Everything about one patient at their snapshot hour: alert, scores
// (with history), confidence, reasons, summary and outcome.
export function getPatient(id) {
  return getJson(`/patients/${id}`);
}

// true if the backend answers GET /health, false if not.
// Used for the "API connected" pill in the header.
export async function checkHealth() {
  try {
    await getJson("/health");
    return true;
  } catch {
    return false;
  }
}
