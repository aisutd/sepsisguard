import { ALERT_LEVELS, levelClass } from "../alertLevels";

// Count tiles above the bed grid: total patients, then one tile per alert level.
// Counts come straight from the alert_level the backend gave each patient.
function AlertCounts({ patients }) {
  return (
    <div className="count-row">
      <div className="count-tile">
        <div className="count-label">Patients</div>
        <div className="count-value">{patients.length}</div>
      </div>

      {ALERT_LEVELS.map((level) => (
        <div key={level} className={`count-tile level-${levelClass(level)}`}>
          <div className="count-label">{level}</div>
          <div className="count-value">
            {patients.filter((p) => p.alert_level === level).length}
          </div>
        </div>
      ))}
    </div>
  );
}

export default AlertCounts;
