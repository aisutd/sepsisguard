// App.jsx: the whole SepsisGuard screen (Maritza owns this file).
//
// How it works:
//   1. On page load (and on Refresh) it fetches the 8 demo patients from
//      GET /patients, then GET /patients/{id} for each one to get their
//      hour-by-hour scores. Patients stay in bed order (demo_patients.json).
//   2. One replay clock (replay.js) drives the Ward, Patient Review and
//      Alerts: every page shows each patient as they were at the clock's hour.
//   3. `page` decides what's shown (see pages.js).
//   4. The selected patient's GET /patients/{id} feeds the Patient Review
//      cards; each teammate's component gets only the part it needs.
// Layout follows the Figma prototype "SepsisGuard App Prototype".
import { useEffect, useState } from "react";

import { getPatient, getPatients } from "./api";
import { REPLAY_START, nextAlertHour, patientAt, useReplayClock } from "./replay";

import AppHeader from "./components/AppHeader";
import DisclaimerBar from "./components/DisclaimerBar";
import ReplayBar from "./components/ReplayBar";
import StatusMessage from "./components/StatusMessage";
import PatientList from "./components/PatientList";
import AlertsPage from "./components/AlertsPage";
import PatientHeader from "./components/PatientHeader";
import RiskScore from "./components/RiskScore";             // Arnav
import ShapChart from "./components/ShapChart";             // Immaculate
import SummaryCard from "./components/SummaryCard";         // Anikait
import RiskHistoryChart from "./components/RiskHistoryChart";
import HowToRead from "./components/HowToRead";
import TrackRecordPage from "./components/TrackRecordPage";
import NavSidebar from "./components/NavSidebar";
import PlaceholderPage from "./components/PlaceholderPage";
import DataSafetyPage from "./components/DataSafetyPage";

import { PAGES } from "./pages";

import "./App.css";

// Pages that follow the replay clock and show its controls
const CLOCK_PAGES = ["ward", "review", "alerts"];

function App() {
  // Ward data
  const [patients, setPatients] = useState([]);       // GET /patients, bed order
  const [histories, setHistories] = useState({});     // { patient_id: hourly scores }
  const [loadingList, setLoadingList] = useState(true);
  const [listError, setListError] = useState(null);

  // Patient Review data
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);         // GET /patients/{id}
  const [detailError, setDetailError] = useState(null);

   // "How to read this" popup
  const [cutoffs, setCutoffs] = useState(null);       // the three alert cutoffs
  const [showHowTo, setShowHowTo] = useState(false);


  // Which page is on screen (ids in pages.js). Starts on the Ward.
  const [page, setPage] = useState("ward");

  // Replay clock: from ICU hour 25 to the latest snapshot hour (113, p003455)
  const lastHour = Math.max(REPLAY_START, ...patients.map((p) => p.snapshot_icu_hour));
  const clock = useReplayClock(lastHour);
  // For "Jump to next alert": the next hour any patient starts alerting
  const nextAlert = nextAlertHour(histories, clock.hour);

  // Sidebar link clicked. Patient Review needs a patient: if none has been
  // picked yet, start with Bed 1.
  function navigate(id) {
    if (id === "review" && selectedId === null && patients.length > 0) {
      setSelectedId(patients[0].patient_id);
    }
    setPage(id);
  }

  // Bed tile or alert row clicked: show that patient on Patient Review
  function openPatient(id) {
    setSelectedId(id);
    setPage("review");
  }

  // Changing this number re-runs both loaders below (the Refresh button)
  const [refreshKey, setRefreshKey] = useState(0);
  const refresh = () => setRefreshKey((k) => k + 1);

  // 1. Load the patients and their hourly scores
  useEffect(() => {
    setLoadingList(true);
    setListError(null);

    getPatients()
      .then(async (list) => {
        // GET /patients has only the snapshot level; the hourly scores are in
        // each patient's details (8 small requests).
        const details = await Promise.all(list.map((p) => getPatient(p.patient_id)));
        setPatients(list);
        setHistories(Object.fromEntries(
          details.map((d) => [d.patient_id, d.scores.available ? d.scores.history : []])
        ));
        // Every patient's details include the cutoffs; read them from the first.
        setCutoffs(details[0]?.alert.cutoffs ?? null);
      })
      .catch((err) => setListError(err.message))
      .finally(() => setLoadingList(false));
  }, [refreshKey]);

  // 2. Load one patient whenever the selection changes
  useEffect(() => {
    if (selectedId === null) return;

    // If the user picks another patient before this answer arrives,
    // ignore this answer so the screen doesn't show the wrong patient.
    let cancelled = false;
    setDetail(null);
    setDetailError(null);

    getPatient(selectedId)
      .then((d) => { if (!cancelled) setDetail(d); })
      .catch((err) => { if (!cancelled) setDetailError(err.message); });

    return () => { cancelled = true; };
  }, [selectedId, refreshKey]);

  // Previous / Next on Patient Review follow the bed order.
  // index is -1 if no patient is selected yet.
  const index = patients.findIndex((p) => p.patient_id === selectedId);
  const goTo = (offset) => {
    const next = patients[index + offset];
    if (next) setSelectedId(next.patient_id);
  };

  // Ward and Alerts both need the patient list first
  let listStatus = null;
  if (loadingList) {
    listStatus = <StatusMessage title="Loading patients…" />;
  } else if (listError) {
    listStatus = <StatusMessage title="Can't reach the backend" message={listError} onRetry={refresh} />;
  }

  // Pick what goes in the main area
  let content;

  if (page === "ward") {
    content = listStatus ?? (
      <>
        <div className="title-row">
          <h1>Ward</h1>
          <p className="muted">
            Eight case studies, chosen on purpose. Press Play to watch the ward change hour by hour.
          </p>
        </div>
        <PatientList patients={patients} histories={histories} cutoffs={cutoffs}
                     hour={clock.hour} onSelect={openPatient} onHowToRead={() => setShowHowTo(true)}/>
      </>
    );
  } else if (page === "alerts") {
    content = listStatus ?? (
      <AlertsPage patients={patients} histories={histories} hour={clock.hour} onSelect={openPatient} />
    );
  } else if (page === "review") {
    // The selected patient as they were at the clock's hour
    const history = detail?.scores?.available ? detail.scores.history : [];
    const now = patientAt(history, clock.hour);

    let body;
    if (selectedId === null) {
      // Happens only if the patient list failed to load
      body = <StatusMessage title="No patient selected" message="Pick a patient on the Ward page." />;
    } else if (detailError) {
      body = <StatusMessage title="Couldn't load this patient" message={detailError} />;
    } else if (!detail) {
      // Data hasn't arrived yet: don't draw the panels until it has
      body = <StatusMessage title="Loading patient…" />;
    } else {
      const snap = detail.snapshot_icu_hour;

      // Horizon cards show the scores at the clock's hour, not the snapshot.
      // Arnav's RiskScore reads scores.at_snapshot, so that's what is swapped.
      let scoresNow = detail.scores;
      if (detail.scores.available) {
        scoresNow = now.entry
          ? { ...detail.scores, at_snapshot: now.entry }
          : { available: false, message: `Not scored yet. The first score is at ICU hour ${history[0]?.icu_hour}.` };
      }

      // Reasons and summary exist only for the snapshot hour, so they're shown
      // only once the clock reaches it. The cards never mix hours.
      const atSnapshot = clock.hour >= snap;

      body = (
        <>
          {/* Onset and "no sepsis" notes appear only once the clock passes them */}
          <RiskHistoryChart scores={detail.scores} alert={detail.alert} hour={clock.hour}
                            onsetHour={detail.sepsis_onset_hour ?? null}
                            stayEnd={detail.total_icu_hours}
                            noSepsis={detail.outcome?.developed_sepsis === false} 
                            onHowToRead={() => setShowHowTo(true)}/>

          <div className="card-row">
            <RiskScore scores={scoresNow} alert={detail.alert} />
            {atSnapshot ? (
              <>
                <ShapChart reasons={detail.reasons} />
                <SummaryCard selectedPatient={detail.patient_id} />
              </>
            ) : (
              <section className="panel snapshot-wait">
                <h2>Reasons and summary</h2>
                <p className="muted">
                  These are worked out at ICU hour {snap}, this patient's snapshot. They appear
                  when the replay reaches that hour, so they always match the scores on screen.
                </p>
                <button className="btn" onClick={() => clock.goTo(snap)}>Go to hour {snap}</button>
              </section>
            )}
          </div>
        </>
      );
    }

    // Patient Review, top to bottom:
    //   1. strip: back to Ward, patient header, Previous / Next
    //   2. chart across the full width
    //   3. three compact cards (horizons, reasons, summary)
    // The page is exactly one screen tall; the cards share the leftover
    // height and scroll inside themselves if needed (see App.css).
    content = (
      <div className="patient-main">
        <div className="panel patient-strip">
          <button className="link-btn" onClick={() => setPage("ward")}>‹ Ward</button>
          {/* Header needs the patient's data; until then just show the ID */}
          {detail ? (
            <PatientHeader detail={detail} level={now.level}
                           hour={Math.min(clock.hour, detail.snapshot_icu_hour)} />
          ) : (
            <span className="strip-info mono">{selectedId}</span>
          )}
          <div className="nav-buttons">
            {/* Position in the bed order, e.g. "3 of 8" */}
            {index !== -1 && (
              <span className="muted small">{index + 1} of {patients.length}</span>
            )}
            <button className="btn btn-small" disabled={index <= 0} onClick={() => goTo(-1)}
                    aria-label="Previous patient">
              ‹ Prev
            </button>
            <button className="btn btn-small" disabled={index === -1 || index >= patients.length - 1}
                    onClick={() => goTo(1)} aria-label="Next patient">
              Next ›
            </button>
          </div>
        </div>
        {body}
      </div>
    );
  } else if (page === "track") {
    content = <TrackRecordPage />;
  } else if (page === "safety") {
    content = <DataSafetyPage />;
  } else {

    // Data and safety: content coming later
    const current = PAGES.find((p) => p.id === page);
    content = <PlaceholderPage title={current?.label ?? "Page"} />;
  }


  return (
    <div className="app">
      <NavSidebar page={page} onNavigate={navigate} />
      {/* fit-screen: on Patient Review the column is exactly one window tall */}
      <div className={`app-main${page === "review" ? " fit-screen" : ""}`}>
        <AppHeader title={PAGES.find((p) => p.id === page)?.label}>
          {CLOCK_PAGES.includes(page) && patients.length > 0 && (
            <ReplayBar clock={clock} nextAlert={nextAlert} />
          )}
        </AppHeader>
        <DisclaimerBar onDetails={() => setPage("safety")} />
        <main className="page">{content}</main>
      </div>
      {showHowTo && <HowToRead cutoffs={cutoffs} onClose={() => setShowHowTo(false)} />}

    </div>
  );
}

export default App;
