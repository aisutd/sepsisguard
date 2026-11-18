// App.jsx: the whole SepsisGuard screen (Maritza owns this file).
//
// How it works:
//   1. On page load (and on Refresh) it fetches the 8 demo patients from
//      GET /patients and sorts them most serious first.
//   2. selectedId decides the screen: null = overview (count tiles + bed grid),
//      a patient_id = that patient's screen.
//   3. When a patient is selected, it fetches GET /patients/{id} and passes each
//      teammate's component only the part of the answer it needs.
// Layout follows the Figma prototype "SepsisGuard App Prototype".
import { useEffect, useState } from "react";

import { checkHealth, getPatient, getPatients } from "./api";
import { sortByAlert } from "./alertLevels";

import AppHeader from "./components/AppHeader";
import DisclaimerBar from "./components/DisclaimerBar";
import StatusMessage from "./components/StatusMessage";
import AlertCounts from "./components/AlertCounts";
import PatientList from "./components/PatientList";
import PatientHeader from "./components/PatientHeader";
import RiskScore from "./components/RiskScore";             // Arnav
import ShapChart from "./components/ShapChart";             // Immaculate
import SummaryCard from "./components/SummaryCard";         // Anikait
import RiskHistoryChart from "./components/RiskHistoryChart";
import HowAlertsWork from "./components/HowAlertsWork";
import PatientSidebar from "./components/PatientSidebar";

import "./App.css";

function App() {
  // Overview data
  const [patients, setPatients] = useState([]);       // sorted GET /patients
  const [loadingList, setLoadingList] = useState(true);
  const [listError, setListError] = useState(null);

  // Patient screen data
  const [selectedId, setSelectedId] = useState(null); // null = overview
  const [detail, setDetail] = useState(null);         // GET /patients/{id}
  const [detailError, setDetailError] = useState(null);

  // Header and popup
  const [apiOk, setApiOk] = useState(null);           // null while checking
  const [cutoffs, setCutoffs] = useState(null);       // for "How alerts work"
  const [showAlerts, setShowAlerts] = useState(false);

  // Changing this number re-runs both loaders below (the Refresh button)
  const [refreshKey, setRefreshKey] = useState(0);
  const refresh = () => setRefreshKey((k) => k + 1);

  // 1. Load the patient list
  useEffect(() => {
    setLoadingList(true);
    setListError(null);
    checkHealth().then(setApiOk);

    getPatients()
      .then((list) => {
        const sorted = sortByAlert(list);
        setPatients(sorted);
        // The backend has no endpoint for just the cutoffs, but every patient
        // detail includes them, so they're read from the first patient.
        if (sorted.length > 0) {
          getPatient(sorted[0].patient_id)
            .then((d) => setCutoffs(d.alert.cutoffs ?? null))
            .catch(() => setCutoffs(null));
        }
      })
      .catch((err) => setListError(err.message))
      .finally(() => setLoadingList(false));
  }, [refreshKey]);

  // 2. Load one patient whenever the selection changes
  useEffect(() => {
    if (selectedId === null) return;

    // If the user clicks another patient before this answer arrives,
    // ignore this answer so the screen doesn't show the wrong patient.
    let cancelled = false;
    setDetail(null);
    setDetailError(null);

    getPatient(selectedId)
      .then((d) => { if (!cancelled) setDetail(d); })
      .catch((err) => { if (!cancelled) setDetailError(err.message); });

    return () => { cancelled = true; };
  }, [selectedId, refreshKey]);

  // Previous / Next follow the same order as the bed grid
  //const index = patients.findIndex((p) => p.patient_id === selectedId);
  //const goTo = (offset) => {
  //  const next = patients[index + offset];
  //  if (next) setSelectedId(next.patient_id);
  //};


  // Pick what goes in the main area
  let content;

  if (selectedId === null) {
    // Overview screen
    if (loadingList) {
      content = <StatusMessage title="Loading patients…" />;
    } else if (listError) {
      content = <StatusMessage title="Can't reach the backend" message={listError} onRetry={refresh} />;
    } else {
      content = (
        <>
          <div className="title-row">
            <h1>ICU patients</h1>
            <p className="muted">
              Demo cohort, sorted by alert level. Select a patient to see risk by horizon, reasons and summary.
            </p>
          </div>
          <AlertCounts patients={patients} />
          <PatientList patients={patients} onSelect={setSelectedId} />
        </>
      );
    }
  } else {
    // Patient screen
    let body;
    if (detailError) {
      body = <StatusMessage title="Couldn't load this patient" message={detailError} />;
    } else if (!detail) {
      body = <StatusMessage title="Loading patient…" />;
    } else {
      body = (
        <>
          <PatientHeader detail={detail} />
          <div className="detail-grid">
            <RiskScore scores={detail.scores} alert={detail.alert} />
            <ShapChart reasons={detail.reasons} />
            {/* key: a fresh chart (and playback state) for every patient */}
            <RiskHistoryChart key={detail.patient_id} scores={detail.scores} alert={detail.alert} />
            <SummaryCard selectedPatient={detail.patient_id} />
          </div>
        </>
      );
    }

    // Patient screen: sidebar on the left, patient details on the right.
    // The sidebar's queue replaces the old Previous / Next buttons.
    content = (
      <div className="patient-layout">
        <PatientSidebar
          patients={patients}
          selectedId={selectedId}
          onSelect={setSelectedId}
          detail={detail}
        />
        <div className="patient-main">
          <button className="link-btn back-link" onClick={() => setSelectedId(null)}>‹ All patients</button>
          {body}
        </div>
      </div>
    );

  }

  return (
    <div className="app">
      <AppHeader apiOk={apiOk} onRefresh={refresh} onShowAlerts={() => setShowAlerts(true)} />
      <DisclaimerBar />
      <main className="page">{content}</main>
      {showAlerts && <HowAlertsWork cutoffs={cutoffs} onClose={() => setShowAlerts(false)} />}
    </div>
  );
}

export default App;
