// Anikait's NLP summary card (Final Sprint, issue #52).
//
// Shows the written summary for the patient picked in the bed grid, at that
// patient's snapshot ICU hour. Usage (Maritza wires it into App.jsx):
//
//   <SummaryCard selectedPatient={patient} />
//
// `selectedPatient` can be a demo patient object (anything with a
// `patient_id`, e.g. an entry from demo_patients.json or GET /patients) or
// just the ID string. With no selection it shows the first demo patient, so
// the card still works on its own.
//
// Data: public/summaries_demo.json, written (together with its twin in
// data/sample/) by `python src/nlp_pipeline/summarize_demo.py`. Each summary
// was rewritten by DistilBART and then checked by verify_summary.py: it may
// name only the patient's top 5 SHAP features, with the right direction
// (raised/lowered) and the right values, and no wording beyond our template's.
// When the AI text fails, it is never shown as the answer: the card shows the
// template summary, which is built straight from the input, and says why.
//
// The "X of 50" line comes from public/summaries_sample.json, the same check
// run over 50 real test patients (verify_summary.py --backend distilbart).
import { useEffect, useState } from "react";

const PASS = { color: "#15803d", background: "#f0fdf4", border: "#bbf7d0" };
const FALLBACK = { color: "#b45309", background: "#fffbeb", border: "#fde68a" };

function loadJson(name) {
  return fetch(`${import.meta.env.BASE_URL}${name}`).then((response) => {
    if (!response.ok) throw new Error(`Could not load ${name}`);
    return response.json();
  });
}

// "3 of 8 failed" text for a list of checked summaries.
function failedText(records) {
  const failed = records.filter((r) => !r.verified).length;
  return `${failed} of ${records.length}`;
}

function SummaryCard({ selectedPatient = null }) {
  const [demo, setDemo] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    loadJson("summaries_demo.json").then(setDemo).catch(() => setError(true));
    // Optional: the card still works if the 50-patient file is missing.
    loadJson("summaries_sample.json").then(setEvaluation).catch(() => setEvaluation(null));
  }, []);

  const selectedId =
    typeof selectedPatient === "string" ? selectedPatient : selectedPatient?.patient_id;
  const record = demo && (selectedId ? demo.find((r) => r.patient_id === selectedId) : demo[0]);

  return (
    <section className="panel" style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      <h2 style={{ margin: 0 }}>
        Prediction summary
        {record && (
          <span style={{ fontWeight: "normal", color: "var(--muted)" }}>
            {" "}· {record.patient_id} · ICU hour {record.icu_hour}
          </span>
        )}
      </h2>

      {error && <p className="placeholder">Could not load the summaries.</p>}
      {!error && !demo && <p className="placeholder">Loading summaries...</p>}
      {demo && !record && (
        <p className="placeholder">No summary for {selectedId}. Summaries exist for the 8 demo patients only.</p>
      )}

      {record && <SummaryBody record={record} />}

      {demo && (
        <p style={{ color: "var(--muted)", fontSize: "0.8rem", margin: 0 }}>
          AI rewrites that failed the check: {failedText(demo)} demo patients
          {evaluation && <>, {failedText(evaluation)} on our 50-patient test set</>}. A failed
          rewrite is replaced by the checked template.
        </p>
      )}

      <p style={{ color: "var(--muted)", fontSize: "0.8rem", margin: 0 }}>
        Reasons come from the {record?.source ?? "XGBoost baseline"} ({record?.horizon ?? 12}h) SHAP
        values. They describe what the model weighed, not a confirmed diagnosis.
      </p>
    </section>
  );
}

// The summary text, its verified/fallback badge, and (on a failure) why.
function SummaryBody({ record }) {
  const tone = record.verified ? PASS : FALLBACK;

  return (
    <>
      <span
        style={{
          alignSelf: "flex-start",
          fontSize: "0.75rem",
          fontWeight: 600,
          padding: "2px 8px",
          borderRadius: "999px",
          color: tone.color,
          background: tone.background,
          border: `1px solid ${tone.border}`,
        }}
      >
        {record.verified ? "✓ Verified AI summary" : "✓ Checked template summary (AI rewrite rejected)"}
      </span>

      {/* Never show failed AI text as the answer: fall back to the template. */}
      <p style={{ lineHeight: 1.6, margin: 0 }}>
        {record.verified ? record.summary : record.fallback_summary}
      </p>

      {!record.verified && (
        <details
          style={{
            fontSize: "0.85rem",
            padding: "0.5rem 0.75rem",
            borderLeft: `3px solid ${tone.color}`,
            background: tone.background,
            lineHeight: 1.5,
          }}
        >
          <summary style={{ cursor: "pointer" }}>
            Why the AI rewrite was rejected: {record.failure_reason}
          </summary>
          <div style={{ color: "var(--muted)", marginTop: "0.4rem" }}>
            Rejected text: "{record.summary}"
          </div>
        </details>
      )}
    </>
  );
}

export default SummaryCard;
