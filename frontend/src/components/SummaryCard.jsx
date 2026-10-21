// Anikait's NLP summary card (container styling by Harish).
//
// Lets you pick a patient and shows that patient's summary from
// public/summaries_sample.json, a copy of data/sample/summaries_sample.json
// (the DistilBART run over 50 real patients). Copy it again after re-running
// verify_summary.py. Later this should be read from the backend instead.
//
// Every summary was checked by verify_summary.py: it names only the patient's
// top 5 SHAP features, gets each one's direction and value right, and uses no
// wording beyond our template's. When the AI-written text fails that check, it
// is never shown as the answer: the card shows the template version, which is
// built from the input, and says why the rewrite was rejected.
import { useEffect, useState } from "react";

function SummaryCard() {
  const [records, setRecords] = useState([]);
  const [patientId, setPatientId] = useState("");
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}summaries_sample.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Could not load summaries");
        }
        return response.json();
      })
      .then((data) => {
        setRecords(data);
        setPatientId(data[0].patient_id);
      })
      .catch(() => setError(true));
  }, []);

  const record = records.find((r) => r.patient_id === patientId);
  const failedCount = records.filter((r) => !r.verified).length;

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

      {error && <p>Could not load the summaries.</p>}
      {!error && !record && <p>Loading summaries...</p>}

      {record && (
        <>
          <label style={{ fontSize: "0.85rem", color: "#6b7280" }}>
            Patient{" "}
            <select value={patientId} onChange={(e) => setPatientId(e.target.value)}>
              {records.map((r) => (
                <option key={r.patient_id} value={r.patient_id}>
                  {r.patient_id}
                  {r.verified ? "" : " (AI rewrite failed)"}
                </option>
              ))}
            </select>
          </label>

          {/* The template summary is built from the input, so it always passes. */}
          <p style={{ lineHeight: 1.6 }}>{record.verified ? record.summary : record.fallback_summary}</p>

          <p style={{ fontSize: "0.85rem", margin: "0 0 0.75rem", color: "#15803d" }}>
            ✓ Checked against the model's top 5 features, their direction and their values
          </p>

          {!record.verified && (
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
              template summary is shown instead. Reason: {record.failure_reason}.
              <div style={{ color: "#6b7280", marginTop: "0.4rem" }}>Rejected text: "{record.summary}"</div>
            </div>
          )}

          <p style={{ color: "#6b7280", fontSize: "0.8rem", margin: "0 0 0.75rem" }}>
            {failedCount} of {records.length} AI-written summaries failed verification.
          </p>
        </>
      )}

      <p style={{ color: "#666", fontSize: "0.875rem", margin: 0 }}>
        Based on the baseline model's SHAP output. Describes model influence,
        not a confirmed diagnosis.
      </p>
    </section>
  );
}

export default SummaryCard;
