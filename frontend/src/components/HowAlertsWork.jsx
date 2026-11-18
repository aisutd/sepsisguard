import AlertBadge from "./AlertBadge";

// "How alerts work" popup from the Figma design, with the real rule from
// backend/main.py: horizons are checked in order and the first score at or
// above its cutoff sets the level.
// cutoffs = { "6h": ..., "12h": ..., "24h": ... } or null while loading.
const RULE = [
  ["6h", "Urgent"],
  ["12h", "High risk"],
  ["24h", "Elevated risk"],
];

function HowAlertsWork({ cutoffs, onClose }) {
  return (
    // Clicking the dark background closes the popup
    <div className="overlay" onClick={onClose}>
      {/* stopPropagation: clicks inside the box don't close it */}
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="alerts-title"
           onClick={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <h2 id="alerts-title">How alerts work</h2>
          <button className="btn-icon" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <p>
          Each patient gets one alert level. The horizons are checked in this order,
          and the first score at or above its cutoff sets the level:
        </p>

        {RULE.map(([horizon, level]) => (
          <div key={horizon} className="modal-row">
            <AlertBadge level={level} />
            <span className="mono small">
              {horizon} score ≥ {cutoffs ? cutoffs[horizon].toFixed(4) : "…"}
            </span>
          </div>
        ))}
        <div className="modal-row">
          <AlertBadge level="No alert" />
          <span className="small">no score at or above its cutoff</span>
        </div>

        <p className="muted small">
          Each cutoff is the score at which 10% of non-sepsis patients ever trigger an alert.
          Scores come from the TFT model and are relative risk, not probabilities.
          Reasons come from the XGBoost baseline.
        </p>
        <button className="btn btn-primary wide" onClick={onClose}>Got it</button>
      </div>
    </div>
  );
}

export default HowAlertsWork;
