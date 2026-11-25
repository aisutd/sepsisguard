import AlertBadge from "./AlertBadge";

// One-line patient header for Patient Review, shown in the strip at the top
// between "‹ Ward" and Previous / Next. Kept to one row so the page fits one
// screen. Shows ID, site, ICU hour, alert level and confidence. The outcome is
// never shown as words: the chart marks sepsis onset only once the replay
// clock passes it (RiskHistoryChart.jsx).

//
// Props:
//   detail  the GET /patients/{id} response
//   level   alert level at the replay clock's hour (App.jsx)
//   hour    the clock's hour, capped at the snapshot
// `level` and `hour` come from the replay clock, so the header matches the
// chart and the horizon cards.
function PatientHeader({ detail, level, hour }) {
  const { confidence } = detail;

  return (
    <div className="strip-info">
      <h1 className="mono strip-id">{detail.patient_id}</h1>
      <AlertBadge level={level} />
      <span className="strip-item">Site {detail.site} · ICU hour {hour} of {detail.snapshot_icu_hour}</span>
            <span className="strip-item">
        Confidence <strong>{confidence.available ? confidence.label : "n/a"}</strong>
      </span>
      {/* Inputs blank in the raw data at this hour, filled in by the cleaning step */}
      {confidence.available && (
        <span className="strip-item">
          {confidence.imputed_count} of {confidence.total_inputs} measurements were missing and estimated
        </span>
      )}

      
    </div>
  );
}

export default PatientHeader;
