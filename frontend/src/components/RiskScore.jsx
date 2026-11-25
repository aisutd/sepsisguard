import AlertBadge from "./AlertBadge";
import { meterFill, thresholdText } from "../alertLevels";
import { SCORES_NOTE } from "../disclaimers";

// Arnav's TFT risk panel ("Sepsis risk by horizon" in the Figma design).
// Props (from GET /patients/{id}):
//   scores  detail.scores: { available, at_snapshot: { "6h", "12h", "24h" }, ... }
//   alert   detail.alert:  { level, cutoffs: { "6h", "12h", "24h" }, ... }
//
// Each card leads with how far the score is past its own horizon's alert
// threshold ("3.6× its alert threshold" = score ÷ cutoff), because the raw
// decimals mean little on their own. The decimals stay as small print.
// Scores are relative risk, so they're never shown as percentages.
// Above the threshold, the badge names the level that horizon would trigger.
const HORIZONS = [
  { key: "6h", label: "Within 6 hours", level: "Urgent" },
  { key: "12h", label: "Within 12 hours", level: "High risk" },
  { key: "24h", label: "Within 24 hours", level: "Elevated risk" },
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
        <span className="muted small">Main model</span>
      </div>

      <div className="horizon-row">
                {HORIZONS.map(({ key, label, level }) => {
          const score = scores.at_snapshot[key];
          const cutoff = alert?.cutoffs?.[key];
          // How many times its threshold the score is (1 = at the threshold)
          const ratio = cutoff ? score / cutoff : null;
          const above = ratio !== null && ratio >= 1;

          return (
            <div key={key} className="horizon-card">
              <div className="horizon-top">
                <span className="horizon-label">{label}</span>
                {above ? <AlertBadge level={level} /> : <span className="badge badge-none">Below</span>}
              </div>
              {/* Bar: the tick at 25% is the threshold (1×); a full bar is 4× */}
              <div className="meter horizon-meter" aria-hidden="true">
                <div className="meter-fill" style={{ width: `${meterFill(ratio)}%` }} />
                <div className="meter-cutoff" />
              </div>
              <div className="horizon-ratio">{thresholdText(ratio)}</div>
              {/* The raw numbers, as small print */}
              <div className="small muted mono">
                score {formatScore(score)} · threshold {formatScore(cutoff)}
              </div>
            </div>
          );
        })}

      </div>
            {/* Arnav's caveats, kept word for word from his original panel. */}
      {/* One short line; Arnav's full caveats are on Data and safety */}
      <p className="panel-foot">{SCORES_NOTE}</p>


    </section>
  );
}

export default RiskScore;
