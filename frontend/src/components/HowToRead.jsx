import AlertBadge from "./AlertBadge";

// "How to read this" popup, opened from the Ward and the risk chart.
// The plain-language version of how scores, bars and levels work.
// (Replaces the old "How alerts work" popup.)
// Props:
//   cutoffs  { "6h", "12h", "24h" } or null while loading. Shown only as
//            small print, because the decimals mean little on their own.
//   onClose  called by ✕, "Got it" or a click outside the box
const RULE = [
  { key: "6h", label: "6-hour", level: "Urgent" },
  { key: "12h", label: "12-hour", level: "High risk" },
  { key: "24h", label: "24-hour", level: "Elevated risk" },
];

function HowToRead({ cutoffs, onClose }) {
  return (
    // Clicking the dark background closes the popup
    <div className="overlay" onClick={onClose}>
      {/* stopPropagation: clicks inside the box don't close it */}
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="howto-title"
           onClick={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <h2 id="howto-title">How to read this</h2>
          <button className="btn-icon" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <p>
          <strong>Color</strong> is the alert level: Urgent, High risk, Elevated risk or
          No alert. The level is always written out too.
        </p>
        <p>
          <strong>The bar</strong> shows how far a score is past its alert threshold. The
          tick marks the threshold (1×), so "2× its alert threshold" means twice the
          score that sets off an alert.
        </p>
        <p>
          <strong>Level:</strong> the 6-hour, then 12-hour, then 24-hour score is checked.
          The first one at its threshold sets the level.
        </p>

        {RULE.map(({ key, label, level }) => (
          <div key={key} className="modal-row">
            <AlertBadge level={level} />
            <span className="small">
              {label} score at its threshold{" "}
              <span className="muted mono">({cutoffs ? cutoffs[key].toFixed(4) : "…"})</span>
            </span>
          </div>
        ))}

        <p>
          <strong>Scores</strong> show how high a patient ranks, not the chance of getting
          sepsis. Compare patients, not horizons.
        </p>
        <button className="btn btn-primary" onClick={onClose}>Got it</button>
      </div>
    </div>
  );
}

export default HowToRead;
