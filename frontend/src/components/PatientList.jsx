import AlertBadge from "./AlertBadge";
import { levelClass } from "../alertLevels";

// Bed grid: one tile per demo patient, from GET /patients.
// Replaces the old table of 6 hardcoded patients from patients_sample.json.
//
// Props:
//   patients  list from the backend, already sorted most serious first (App.jsx)
//   onSelect  called with a patient_id when a tile is clicked
//
// Each tile shows the alert level in words (AlertBadge) and with a colored
// left edge, so it reads with or without color.
function PatientList({ patients, onSelect }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Patients</h2>
        <span className="muted small">Sorted by alert level · click a patient for details</span>
      </div>

      <div className="bed-grid">
        {patients.map((p) => (
          <button
            key={p.patient_id}
            className={`bed-tile level-${levelClass(p.alert_level)}`}
            onClick={() => onSelect(p.patient_id)}
          >
            <div className="bed-top">
              <span className="mono bed-id">{p.patient_id}</span>
              <AlertBadge level={p.alert_level} />
            </div>
            <div className="bed-meta">Site {p.site} · ICU hour {p.snapshot_icu_hour}</div>
            <div className="bed-meta">Confidence: <strong>{p.confidence}</strong></div>
            {/* The real outcome. Labelled "demo only": a real ICU wouldn't know it yet. */}
            <div className="bed-meta">
              Outcome (demo only): <strong>{p.outcome.developed_sepsis ? "Sepsis" : "No sepsis"}</strong>
            </div>
          </button>
        ))}
      </div>

      <p className="panel-foot">
        Alert levels use the cutoffs in alert_thresholds.json. The outcome is shown for the demo only.
      </p>
    </section>
  );
}

export default PatientList;
