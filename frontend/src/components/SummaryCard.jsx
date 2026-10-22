// Anikait's NLP summary card (container styling by Harish).
//
// Shows one summary from data/sample/summaries_sample.json that passed
// verify_summary.py (it names only this patient's top 5 SHAP features, gets
// each one's direction right, and uses no wording beyond our template's).
//
// Hardcoded copy for now, per Issue #26. It is a real output: one row of
// data/sample/shap_top_features.json (the baseline model's SHAP export) run
// through generate_summary.py and verify_summary.py. Later this should be
// read from the backend instead.
const summaryRecord = {
  patient_id: "p000006",
  summary:
    "Time in the ICU raised this patient's predicted sepsis risk the most, followed by heart rate, respiratory rate, and lactate. Temperature lowered it. These reflect what the model weighed, not measured values or a diagnosis.",
  verified: true,
};

function SummaryCard() {
  const { patient_id, summary, verified } = summaryRecord;

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

      <p style={{ lineHeight: 1.6 }}>{summary}</p>

      {/* Only verified summaries should ever reach this card, but show the
          flag anyway so a failure would be visible rather than silent. */}
      <p style={{ fontSize: "0.85rem", margin: "0 0 0.75rem", color: verified ? "#15803d" : "#b91c1c" }}>
        {verified ? "✓ Checked against the model's top 5 features" : "✗ Failed verification, do not use"}
        <span style={{ color: "#6b7280" }}> · Patient {patient_id}</span>
      </p>

      <p style={{ color: "#666", fontSize: "0.875rem", margin: 0 }}>
        Based on the baseline model's SHAP output. Describes model influence,
        not a confirmed diagnosis.
      </p>
    </section>
  );
}

export default SummaryCard;
