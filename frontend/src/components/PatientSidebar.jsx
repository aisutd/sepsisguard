import AlertBadge from "./AlertBadge";
import { levelClass } from "../alertLevels";

// Sidebar on the patient screen, for clinicians. Answers three questions
// without extra clicks:
//   1. Patient queue: who else needs attention? (click to switch patient)
//   2. Why this alert: which score crossed which cutoff?
//
// Props:
//   patients    sorted list from GET /patients (most serious first, App.jsx)
//   selectedId  the patient on screen
//   onSelect    called with a patient_id when a queue item is clicked
//   detail      GET /patients/{id} response, or null while it loads
//
// Everything comes from data App.jsx already loads; the backend is unchanged.

const fmt = (n) => (typeof n === "number" ? n.toFixed(4) : "n/a");

function PatientSidebar({ patients, selectedId, onSelect, detail }) {
  return (
    <aside className="sidebar">
      <section className="panel">
        <h2>Patient queue</h2>
        <ul className="queue">
          {patients.map((p) => {
            const selected = p.patient_id === selectedId;
            return (
              <li key={p.patient_id}>
                <button
                  className={`queue-item level-${levelClass(p.alert_level)}${selected ? " selected" : ""}`}
                  onClick={() => onSelect(p.patient_id)}
                  aria-current={selected ? "true" : undefined}
                >
                  <span className="mono">{p.patient_id}</span>
                  <AlertBadge level={p.alert_level} />
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      {/* Needs the patient's detail, so it waits until it has loaded */}
      {detail && <AlertReason alert={detail.alert} scores={detail.scores} />}
    </aside>
  );
}

// One sentence on why the patient has their alert level.
// The backend's alert.triggered_by names the horizon that set the level
// (null when no score is at or above its cutoff).
function AlertReason({ alert, scores }) {
  let text;
  if (!scores?.available || !alert?.cutoffs) {
    text = alert?.message ?? scores?.message ?? "Alert reason not available.";
  } else if (alert.triggered_by) {
    const h = alert.triggered_by;
    text = (
      <>
        The <strong>{h}</strong> score (<span className="mono">{fmt(scores.at_snapshot[h])}</span>) is
        at or above its cutoff (<span className="mono">{fmt(alert.cutoffs[h])}</span>).
      </>
    );
  } else {
    text = "All three scores are below their cutoffs.";
  }

  return (
    <section className="panel">
      <h2>Why this alert</h2>
      <div className="sidebar-badge"><AlertBadge level={alert.level} /></div>
      <p className="small">{text}</p>
      <p className="muted small">Checked in order 6h → 12h → 24h; the first score at or above its cutoff sets the level.</p>
    </section>
  );
}


export default PatientSidebar;
