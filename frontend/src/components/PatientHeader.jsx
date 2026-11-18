import AlertBadge from "./AlertBadge";

// Header of the patient screen (checklist: ID, site, ICU hour, confidence,
// outcome labelled "demo only"). `detail` is the GET /patients/{id} response.
function PatientHeader({ detail }) {
  const { alert, confidence, outcome } = detail;

  return (
    <section className="panel patient-header">
      <div>
        <div className="patient-title">
          <h1 className="mono">{detail.patient_id}</h1>
          <AlertBadge level={alert.level} />
        </div>
        <div className="muted small">
          Site {detail.site} · ICU hour {detail.snapshot_icu_hour} of {detail.total_icu_hours}
          {/* Which horizon's score set the level, e.g. "set by the 6h score" */}
          {alert.triggered_by && <> · level set by the {alert.triggered_by} score</>}
        </div>
      </div>

      <div className="header-stats">
        <div>
          <div className="stat-label">Confidence</div>
          <div className="stat-value">{confidence.available ? confidence.label : "not available"}</div>
          {/* Confidence depends on how many inputs were filled in instead of measured */}
          {confidence.available && (
            <div className="muted small">
              {confidence.imputed_count} of {confidence.total_inputs} inputs filled in
            </div>
          )}
        </div>
        <div>
          <div className="stat-label">Outcome (demo only)</div>
          <div className="stat-value">{outcome.developed_sepsis ? "Sepsis" : "No sepsis"}</div>
        </div>
      </div>
    </section>
  );
}

export default PatientHeader;
