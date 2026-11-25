import AlertBadge from "./AlertBadge";
import { levelClass, meterFill } from "../alertLevels";
import { isAlert, levelRatio, patientAt } from "../replay";

// Ward bed grid, driven by the shared replay clock (App.jsx).
//
// Props:
//   patients   GET /patients in bed order (Bed 1 = first). The order never
//              changes; only the colors and words do, so beds are easy to follow.
//   histories  { patient_id: hourly scores } from GET /patients/{id}
//   cutoffs    the three alert cutoffs (for the "6-hour score: 7.1× its threshold" line)
//   hour       the clock's ICU hour
//   onSelect   called with a patient_id when a tile is clicked
//  onHowToRead opens the "How to read this" popup
// Each tile shows the level in words (AlertBadge) and as a colored left edge,
// so it reads with or without color.
function PatientList({ patients, histories, cutoffs, hour, onSelect, onHowToRead }) {
  const beds = patients.map((p) => ({ p, ...patientAt(histories[p.patient_id], hour) }));
  const alertCount = beds.filter((b) => isAlert(b.level)).length;

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{alertCount === 1 ? "1 alert now" : `${alertCount} alerts now`}</h2>
        <span className="muted small">
          ICU hour {hour} ·{" "}
          <button className="link-btn" onClick={onHowToRead}>How to read this</button>
        </span>

      </div>

      <div className="bed-grid">
        {beds.map(({ p, status, entry, level, lastHour, firstAlert }, i) => {
          // The score that set the level (e.g. Urgent = the 6-hour score), so
          // the number always matches the badge
          const lr = levelRatio(entry, cutoffs);
          const ratio = lr?.ratio ?? null;
          // Bar: the mark at 25% is the threshold, so a full bar = 4× the threshold
          const fill = meterFill(ratio);

          // Line 1: that score compared with its own threshold
          let scoreText;
          if (status === "waiting") scoreText = "Not scored yet";
          else if (ratio === null) scoreText = "Score not available";
          else if (ratio < 1) scoreText = "Below its alert thresholds";
          else scoreText = `${parseInt(lr.horizon)}-hour score: ${ratio.toFixed(1)}× its threshold`;

          // Line 2: first alert so far, or "frozen" once their scores end
          const parts = [];
          if (status === "ended") parts.push(`Last scored at hour ${lastHour}`);
          if (firstAlert !== null) parts.push(`First alert at hour ${firstAlert}`);
          if (parts.length === 0) parts.push(status === "waiting" ? "Waiting for enough data" : "No alert so far");

          return (
            <button
              key={p.patient_id}
              className={`bed-tile level-${levelClass(level)}${status === "ended" ? " ended" : ""}`}
              onClick={() => onSelect(p.patient_id)}
            >
              <div className="bed-top">
                <span className="bed-id">Bed {i + 1}</span>
                <AlertBadge level={level} />
              </div>
              <div className="bed-meta"><span className="mono">{p.patient_id}</span> · Site {p.site}</div>
              <div className="meter bed-meter" aria-hidden="true">
                <div className="meter-fill" style={{ width: `${fill}%` }} />
                <div className="meter-cutoff" />
              </div>
              <div className="bed-meta">{scoreText}</div>
              <div className="bed-meta muted">{parts.join(" · ")}</div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export default PatientList;
