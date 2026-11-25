import { useEffect, useState } from "react";

// Data and safety: the full explanations behind the short lines on the other
// pages (disclaimers.js). The banner's "Details" link opens this page.
//
// The summary pass counts are read from the same files SummaryCard uses
// (public/summaries_demo.json and summaries_sample.json, written by
// src/nlp_pipeline/summarize_demo.py), so they stay correct if the summaries
// are rebuilt. Other numbers come from docs/final_results.md and
// backend/confidence.py.

// Load one file from frontend/public; null if it's missing
const loadPublic = (name) =>
  fetch(`${import.meta.env.BASE_URL}${name}`)
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);

// "2 of 8" summaries where the AI rewrite was rejected
const failedText = (rows) =>
  rows ? `${rows.filter((r) => !r.verified).length} of ${rows.length}` : "…";

function DataSafetyPage() {
  const [summaries, setSummaries] = useState({ demo: null, sample: null });

  useEffect(() => {
    Promise.all([loadPublic("summaries_demo.json"), loadPublic("summaries_sample.json")])
      .then(([demo, sample]) => setSummaries({ demo, sample }));
  }, []);

  return (
    <>
      <div className="title-row">
        <h1>Data and safety</h1>
        <p className="muted">What the screens show, where it comes from, and what it can't tell you.</p>
      </div>

      <div className="safety-grid">
        <section className="panel">
          <h2>What this is</h2>
          <p>
            A research system built on historical ICU data from two hospitals (the PhysioNet 2019
            sepsis challenge). It is not for clinical use and has not been tested in a live ICU.
          </p>
          <p>
            The replay shows each patient hour by hour. Nothing after the current replay hour is
            ever shown.
          </p>
        </section>

        <section className="panel">
          <h2>Scores</h2>
          <p>
            Scores come from our main model (a Temporal Fusion Transformer). They rank patients:
            a higher score means higher risk than other patients, not a chance of getting sepsis.
          </p>
          <p>
            Longer prediction windows score higher because more hours count as "before sepsis",
            so compare patients within the same window, not scores across windows. These are
            experimental predictions and have not been validated for clinical use.
          </p>
        </section>

        <section className="panel">
          <h2>Alerts</h2>
          <p>
            Each window has its own alert threshold, set so that 10% of patients without sepsis
            get an alert from it. All three together alert on about 1 in 7 patients without
            sepsis (14.5%). See Track record for how many sepsis patients were warned in time.
          </p>
        </section>

        <section className="panel">
          <h2>Sepsis onset</h2>
          <p>
            The onset marker is the hour sepsis was recorded in the dataset (the challenge marks
            patients 6 hours before onset, so onset = first marked hour + 6). It was decided
            after the fact. The model never sees it, and it appears only once the replay passes
            that hour.
          </p>
        </section>

        <section className="panel">
          <h2>Reasons</h2>
          <p>
            Reasons come from our baseline model (XGBoost, 12-hour window), which is easier to
            explain than the main model. Each bar shows how much one measurement pushed this
            patient's baseline score up or down (SHAP values).
          </p>
          <p>
            They are a guide to what drives risk, not an exact account of the main model's score,
            and not a diagnosis or a cause.
          </p>
        </section>

        <section className="panel">
          <h2>Summaries</h2>
          <p>
            An AI model rewrites the reasons as a sentence. A checker then makes sure it names
            only the patient's top 5 reasons, with the right direction and values. If it fails,
            the screen shows a checked template sentence instead, marked "Checked template used".
          </p>
          <p>
            AI rewrites rejected: {failedText(summaries.demo)} demo patients
            {summaries.sample && <>, {failedText(summaries.sample)} on our 50-patient test set</>}.
          </p>
        </section>

        <section className="panel">
          <h2>Missing measurements</h2>
          <p>
            Not every measurement is taken every hour. "22 of 35 measurements were missing and
            estimated" means 22 values were blank at that hour and were filled in with the 
            patient's last measured value, or a typical value if it was never measured. 
            Confidence is high when under 10% are estimated, low when over 40%, and medium in between.
          </p>
        </section>

        <section className="panel">
          <h2>The 8 patients</h2>
          <p>
            Case studies picked by a fixed rule before looking at any model output: 4 who developed
            sepsis and 4 who didn't, drawn at random. In the full test set about 6 in 100 patients
            developed sepsis, so the Ward is not a typical ward.
          </p>
        </section>
      </div>
    </>
  );
}

export default DataSafetyPage;
