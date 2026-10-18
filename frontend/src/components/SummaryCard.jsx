// Actual output from generate_summary.py, mock example 1.
const generatedSummary =
  "The prediction model identified these leading contributors to its sepsis risk prediction: lactate, respiratory rate, heart rate.";

function SummaryCard() {
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

      <p style={{ lineHeight: 1.6 }}>{generatedSummary}</p>

      <p style={{ color: "#666", fontSize: "0.875rem" }}>
        Demo using mock feature scores. Describes model influence, not a
        confirmed diagnosis.
      </p>
    </section>
  );
}

export default SummaryCard;
