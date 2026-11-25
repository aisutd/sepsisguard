import { featureLabel } from "../featureNames";
import { REASONS_NOTE } from "../disclaimers";

// Feature names are shown in plain words (featureNames.js); the data keeps the codes.

// Immaculate's reasons panel ("Why this score" in the Figma design).
// Props: reasons = detail.reasons from GET /patients/{id}:
//   { available, horizon, top_features: [{ feature_name, shap_value, value }] }
//   or { available: false, message } when no SHAP file has this patient
//   at their snapshot hour yet.
// Positive SHAP value = raised the risk (orange), negative = lowered it (blue).
function ShapChart({ reasons }) {
  if (!reasons?.available) {
    return (
      <section className="panel">
        <h2>Why this score</h2>
        <p className="muted">{reasons?.message ?? "Reasons not available."}</p>
      </section>
    );
  }

  const features = reasons.top_features;
  // Longest bar = the feature with the biggest effect; others scale to it
  const biggest = Math.max(...features.map((f) => Math.abs(f.shap_value)));

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Why this score</h2>
        <div className="legend small">
          <span><span className="swatch raises" />Raises risk</span>
          <span><span className="swatch lowers" />Lowers risk</span>
        </div>
      </div>

      {features.map((f) => {
        const raises = f.shap_value > 0;
        return (
          <div key={f.feature_name} className="reason">
            <div className="reason-top">
              {/* Plain name on screen; hovering shows the original column code */}
              <span title={f.feature_name}>
                {featureLabel(f.feature_name)}
                {f.value != null && <span className="muted small"> {f.value}</span>}
              </span>

              <span className="mono">{raises ? "+" : ""}{f.shap_value.toFixed(4)}</span>
            </div>
            <div className="reason-bar">
              <div className={`reason-fill ${raises ? "raises" : "lowers"}`}
                   style={{ width: `${(Math.abs(f.shap_value) / biggest) * 100}%` }} />
            </div>
          </div>
        );
      })}

      {/* One label; the XGBoost / SHAP explanation is on Data and safety */}
      <p className="panel-foot">{REASONS_NOTE}</p>

    </section>
  );
}

export default ShapChart;
