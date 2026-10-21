// Anikait's NLP summary card (container styling by Harish).
//
// Shows one summary from data/sample/summaries_sample.json for a real patient.
// Every summary is checked by verify_summary.py: it names only this patient's
// top 5 SHAP features, gets each one's direction and value right, and uses no
// wording beyond our template's.
//
// This is a real failure kept on purpose: DistilBART wrote the text in
// "summary", it failed the direction check, and the card shows the template
// version instead and says why. A failed summary is never shown as the answer.
//
// Hardcoded copy of the record for p000060 (from the DistilBART run over 50
// real patients). Later this should be read from the backend instead.
const summaryRecord = {
  patient_id: "p000060",
  summary:
    "Time between hospital and ICU admission (-0.02) raised this patient's predicted sepsis risk the most. Temperature (37.67), blood urea nitrogen (36), and white blood cell count (21.7) lowered it. These reflect what the model weighed, based on the values it was given, not a diagnosis.",
  verified: false,
  failure_reason:
    "says 'Temperature' lowered the risk, but its SHAP value is positive; says 'blood urea nitrogen' lowered the risk, but its SHAP value is positive; says 'white blood cell count' lowered the risk, but its SHAP value is positive",
  fallback_summary:
    "Time between hospital and ICU admission (-0.02) raised this patient's predicted sepsis risk the most, followed by temperature (37.67), blood urea nitrogen (36), and white blood cell count (21.7). Fraction of inspired oxygen (0.4) lowered it. These reflect what the model weighed, based on the values it was given, not a diagnosis.",
};

function SummaryCard() {
  const { patient_id, summary, verified, failure_reason, fallback_summary } = summaryRecord;

  // The template summary is built from the input, so it always passes.
  const shown = verified ? summary : fallback_summary;

  return (
    <section
      style={{
        marginTop: "2rem",
        padding: "1.5rem",
        border: "1px solid #d1d5db",
        borderRadius: "12px",
        maxWidth: "650px",
      }}
    >
      <h2 style={{ marginTop: 0 }}>Prediction summary</h2>

      <p style={{ lineHeight: 1.6 }}>{shown}</p>

      <p style={{ fontSize: "0.85rem", margin: "0 0 0.75rem", color: "#15803d" }}>
        ✓ Checked against the model's top 5 features, their direction and their values
        <span style={{ color: "#6b7280" }}> · Patient {patient_id}</span>
      </p>

      {!verified && (
        <div
          style={{
            fontSize: "0.85rem",
            margin: "0 0 0.75rem",
            padding: "0.6rem 0.8rem",
            borderLeft: "3px solid #b91c1c",
            background: "#fef2f2",
            lineHeight: 1.5,
          }}
        >
          <strong>The AI-written summary failed verification</strong>, so the checked
          template summary is shown instead. Reason: {failure_reason}.
          <div style={{ color: "#6b7280", marginTop: "0.4rem" }}>Rejected text: "{summary}"</div>
        </div>
      )}

      <p style={{ color: "#666", fontSize: "0.875rem", margin: 0 }}>
        Based on the baseline model's SHAP output. Describes model influence,
        not a confirmed diagnosis.
      </p>
    </section>
  );
}

export default SummaryCard;
