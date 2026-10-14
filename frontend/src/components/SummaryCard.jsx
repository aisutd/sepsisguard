// Anikait's NLP summary card.
//
// Shows one summary from data/sample/summaries_sample.json that passed
// verify_summary.py (it names only this patient's top 5 SHAP features, gets
// each one's direction right, and makes no diagnostic claim).
//
// Hardcoded copy for now, per Issue #26. It came from the mock SHAP input;
// replace it with a real patient's output once data/sample/shap_top_features.json
// lands, and later read it from the backend instead.
const summaryRecord = {
  patient_id: "mock_p001",
  summary:
    "Lactate raised this patient's predicted sepsis risk the most, followed by time in the ICU and heart rate. Temperature and age lowered it. These reflect what the model weighed, not measured values or a diagnosis.",
  verified: true,
};

function SummaryCard() {
  const { patient_id, summary, verified } = summaryRecord;

  return (
    <section className="panel">
      <h2>Prediction Summary</h2>

      <p style={{ lineHeight: 1.6, margin: "0 0 0.75rem" }}>{summary}</p>

      {/* Only verified summaries should ever reach this card, but show the
          flag anyway so a failure would be visible rather than silent. */}
      <p style={{ fontSize: "0.85rem", margin: 0, color: verified ? "#15803d" : "#b91c1c" }}>
        {verified ? "✓ Checked against the model's top 5 features" : "✗ Failed verification, do not use"}
        <span style={{ color: "#6b7280" }}> · Patient {patient_id} (mock data)</span>
      </p>
    </section>
  );
}

export default SummaryCard;
