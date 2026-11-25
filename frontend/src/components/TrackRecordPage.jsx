// Track record: how the models did on the full test set, in plain words.
//
// Every number is copied from docs/final_results.md (the team's final
// evaluation, src/evaluation/final_evaluation.py). If that script is re-run
// and the numbers change, update them here.
//
// "Warned" = the first alert came at least 6 hours before the recorded sepsis
// onset, using the app's alert setting: each horizon's cutoff alerts on 10% of
// patients without sepsis.

const TEST_SET = { patients: 5972, sepsis: 357 };
const ANY_ALERT_NO_SEPSIS = 14.5;   // % of non-sepsis patients alerted by any of the 3 horizons

// Sepsis patients warned 6+ hours ahead, out of 357: [count, percent]
const COMPARISON = [
  { horizon: "Within 6 hours", main: [186, 52.1], baseline: [199, 55.7] },
  { horizon: "Within 12 hours", main: [191, 53.5], baseline: [198, 55.5] },
  { horizon: "Within 24 hours", main: [193, 54.1], baseline: [193, 54.1] },
];

// Main model, % of each site's sepsis patients warned 6+ hours ahead, by horizon (6h, 12h, 24h).
// Cutoffs here were set within each site.
const SITES = [
  { site: "A", patients: 3086, sepsis: 232, warned: [54.3, 55.2, 56.0] },
  { site: "B", patients: 2886, sepsis: 125, warned: [49.6, 50.4, 52.0] },
];

const per100 = (part, whole) => Math.round((part / whole) * 100);
const fmt = (n) => n.toLocaleString("en-US");

function TrackRecordPage() {
  return (
    <>
      <div className="title-row">
        <h1>Track record</h1>
        {/* 5,972 = test patients the main model could score (25+ hours in the ICU),
            not the whole test set (8,067), so "about 6 in 100" is for this group */}
        <p className="muted">
          How the models did on {fmt(TEST_SET.patients)} past ICU patients who stayed at least
          25 hours. The models never saw them while learning.
        </p>

      </div>

      {/* Headline numbers */}
      <div className="stat-row">
        <div className="stat-tile">
          <div className="stat-value">About {per100(TEST_SET.sepsis, TEST_SET.patients)} in 100</div>
          <p>
            patients developed sepsis ({TEST_SET.sepsis} of {fmt(TEST_SET.patients)}). The 8 on the
            Ward are case studies chosen on purpose, not a typical ward.
          </p>
        </div>
        <div className="stat-tile">
          <div className="stat-value">About 1 in 2</div>
          <p>
            patients with sepsis got an alert at least 6 hours before onset
            ({COMPARISON[0].main[0]}–{COMPARISON[2].main[0]} of {TEST_SET.sepsis}, main model).
          </p>
        </div>
        <div className="stat-tile">
          <div className="stat-value">About 1 in 7</div>
          <p>
            patients without sepsis got at least one alert during their stay
            ({ANY_ALERT_NO_SEPSIS}%). These are false alarms.
          </p>
        </div>
      </div>

      {/* Model comparison */}
      <section className="panel">
        <div className="panel-head">
          <h2>Our main model vs the baseline</h2>
          <span className="muted small">Sepsis patients warned at least 6 hours ahead</span>
        </div>
        <table className="data-table">
          <thead>
            <tr><th>Prediction window</th><th>Main model</th><th>Baseline model</th></tr>
          </thead>
          <tbody>
            {COMPARISON.map((r) => (
              <tr key={r.horizon}>
                <td>{r.horizon}</td>
                <td>{r.main[1]}% <span className="muted small">({r.main[0]} of {TEST_SET.sepsis})</span></td>
                <td>{r.baseline[1]}% <span className="muted small">({r.baseline[0]} of {TEST_SET.sepsis})</span></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="panel-foot">
          The two models do about the same. The baseline warns a few more patients for the 6- and
          12-hour windows (13 and 7 more) and they tie at 24 hours. The app uses the main model for
          scores and the baseline model for reasons.
        </p>
      </section>

      {/* Per site */}
      <section className="panel">
        <div className="panel-head">
          <h2>By hospital</h2>
          <span className="muted small">Main model · sepsis patients warned at least 6 hours ahead</span>
        </div>
        <table className="data-table">
          <thead>
            <tr><th>Hospital</th><th>Patients</th><th>Had sepsis</th><th>6-hour</th><th>12-hour</th><th>24-hour</th></tr>
          </thead>
          <tbody>
            {SITES.map((s) => (
              <tr key={s.site}>
                <td>Site {s.site}</td>
                <td>{fmt(s.patients)}</td>
                <td>{s.sepsis} <span className="muted small">(about {per100(s.sepsis, s.patients)} in 100)</span></td>
                {s.warned.map((w, i) => <td key={i}>{w}%</td>)}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="panel-foot">
          The pattern holds at both hospitals. Site B has fewer sepsis patients, so a few patients
          move its percentages more. Here each hospital's cutoffs were set within that hospital.
        </p>
      </section>

      {/* Caveats */}
      <section className="panel">
        <h2>Read with care</h2>
        <ul className="caveats">
          <li>
            Only hours the main model could score are counted (after each patient's first 24 hours),
            so these numbers are higher than they would be for a whole stay. Compare them with each
            other, not with other studies.
          </li>
          <li>
            Warnings are often long (about 40 hours on average), mostly because some patients look
            high-risk early in their stay. That isn't precise timing of when sepsis starts.
          </li>
          <li>Historical data from two hospitals. Not tested in a live ICU.</li>
        </ul>
      </section>
    </>
  );
}

export default TrackRecordPage;
