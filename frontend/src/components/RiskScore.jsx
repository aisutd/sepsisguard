import AlertBadge from "./AlertBadge";

// Arnav's TFT risk panel ("Sepsis risk by horizon" in the Figma design).
// Props (from GET /patients/{id}):
//   scores  detail.scores: { available, at_snapshot: { "6h", "12h", "24h" }, ... }
//   alert   detail.alert:  { level, cutoffs: { "6h", "12h", "24h" }, ... }
//
// Scores are relative risk, not probabilities, so they're shown as plain
// numbers (4 decimals), never as percentages.
// Each horizon has ONE cutoff, so a score is either above or below it.
// Above the cutoff, the badge names the level that horizon would trigger.
const HORIZONS = [
  { key: "6h", label: "Next 6h", level: "Urgent" },
  { key: "12h", label: "Next 12h", level: "High risk" },
  { key: "24h", label: "Next 24h", level: "Elevated risk" },
];

const formatScore = (n) => (typeof n === "number" ? n.toFixed(4) : "n/a");

function RiskScore({ scores, alert }) {
  if (!scores?.available) {
    return (
      <section className="panel">
        <h2>Sepsis risk by horizon</h2>
        <p className="muted">{scores?.message ?? "Scores not available."}</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Sepsis risk by horizon</h2>
        <span className="muted small">TFT model · relative risk</span>
      </div>

      <div className="horizon-row">
        {HORIZONS.map(({ key, label, level }) => {
          const score = scores.at_snapshot[key];
          const cutoff = alert?.cutoffs?.[key];
          const above = cutoff !== undefined && score >= cutoff;
          // Bar: the black mark in the middle is the cutoff, so a half-full bar
          // means "at the cutoff". Capped at full.
          const fill = cutoff ? Math.min(score / (2 * cutoff), 1) * 100 : 0;

          return (
            <div key={key} className="horizon-card">
              <div className="muted small">{label}</div>
              <div className="horizon-score mono">{formatScore(score)}</div>
              {cutoff !== undefined && (
                <>
                  <div className="meter" aria-hidden="true">
                    <div className="meter-fill" style={{ width: `${fill}%` }} />
                    <div className="meter-cutoff" />
                  </div>
                  <div className="small muted">cutoff <span className="mono">{formatScore(cutoff)}</span></div>
                </>
              )}
              {above
                ? <AlertBadge level={level} />
                : <span className="badge badge-none">Below cutoff</span>}
            </div>
          );
        })}
      </div>
            {/* Arnav's caveats, kept word for word from his original panel. */}
      <p className="panel-foot">
        Scores are relative risk, not calibrated probabilities. Longer horizons can
        score higher because more hours are labeled as sepsis. Compare patients
        within the same horizon, not scores across horizons.
      </p>

      <p className="panel-foot">
        These are experimental TFT predictions, not live patient monitoring. They
        have not been validated for clinical use.
      </p>

    </section>
  );
}

export default RiskScore;
