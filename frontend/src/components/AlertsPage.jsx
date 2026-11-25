import AlertBadge from "./AlertBadge";
import { ALERT_LEVELS, levelClass } from "../alertLevels";
import { isAlert, patientAt } from "../replay";

// Alerts page: every patient with an alert at the clock's hour, most serious
// first. It follows the same replay clock as the Ward (App.jsx).
// Clicking a row opens that patient on Patient Review.
function AlertsPage({ patients, histories, hour, onSelect }) {
  const rows = patients
    .map((p, i) => ({ p, bed: i + 1, ...patientAt(histories[p.patient_id], hour) }))
    .filter((r) => isAlert(r.level))
    .sort((a, b) => ALERT_LEVELS.indexOf(a.level) - ALERT_LEVELS.indexOf(b.level));

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Alerts at ICU hour {hour}</h2>
        <span className="muted small">Most serious first · click to review</span>
      </div>

      {rows.length === 0 ? (
        <p className="muted">No alerts at this hour.</p>
      ) : (
        <ul className="alert-list">
          {rows.map(({ p, bed, level, firstAlert, status, lastHour }) => (
            <li key={p.patient_id}>
              <button className={`alert-row level-${levelClass(level)}`} onClick={() => onSelect(p.patient_id)}>
                <AlertBadge level={level} />
                <span>Bed {bed} · <span className="mono">{p.patient_id}</span> · Site {p.site}</span>
                <span className="muted small">
                  First alert at hour {firstAlert}
                  {status === "ended" && ` · last scored at hour ${lastHour}`}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default AlertsPage;
