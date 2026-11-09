
import { useEffect, useState } from "react";

// Historical TFT predictions for frontend testing.
// These should be updated with final locked-model scores.
const examples = [
    {
        site: "A",
        patient_id: "p007837",
        ICULOS: 116,
        developed_sepsis: true,
        risk_score_6h: 0.2125,
        risk_score_12h: 0.4274,
        risk_score_24h: 0.7449,
    },
    {
        site: "B",
        patient_id: "p102981",
        ICULOS: 33,
        developed_sepsis: false,
        risk_score_6h: 0.003162,
        risk_score_12h: 0.0038,
        risk_score_24h: 0.0054,
    },

    {
        site: "B",
        patient_id: "p113489",
        ICULOS: 52,
        developed_sepsis: true,
        risk_score_6h: 0.0508,
        risk_score_12h: 0.0424,
        risk_score_24h: 0.1653,
    },
    {
        site: "B",
        patient_id: "p113981",
        ICULOS: 37,
        developed_sepsis: false,
        risk_score_6h: 0.0035,
        risk_score_12h: 0.0038,
        risk_score_24h: 0.0061,
    },

];


const horizons = [
    { key: "6h", label: "Urgent" },
    { key: "12h", label: "High Risk" },
    { key: "24h", label: "Elevated Risk" },
];

function RiskScore({ selectedPatient = null, riskData = null }) {
    const [stats, setStats] = useState(null);
    const [error, setError] = useState(false);
    const [exampleIndex, setExampleIndex] = useState(0);

    // Use the dashboard's selected patient when provided.
    // Otherwise, use a local example for testing.
    const patient = selectedPatient ?? examples[exampleIndex];

    const displayHour =
        patient?.ICULOS ?? patient?.hours_in_icu ?? "Unavailable";

    // The overall alert must come from the backend.
    const overallAlert =
        riskData?.overall_alert_level ?? "Awaiting backend";

    useEffect(() => {
        fetch(`${import.meta.env.BASE_URL}pipeline-status.json`)
            .then((response) => {
                if (!response.ok) {
                    throw new Error("Could not load pipeline status");
                }
                return response.json();
            })
            .then(setStats)
            .catch(() => setError(true));
    }, []);

    // Format model scores without treating missing data as zero.
    function formatScore(score) {
        if (typeof score !== "number" || !Number.isFinite(score)) {
            return "Unavailable";
        }

        return `${(score * 100).toFixed(2)}%`;
    }

    return (
        <section className="panel">
            <h2>Risk Score</h2>

            <h3>Patient Risk Assessment</h3>

            {/* Local example selector until dashboard integration */}
            {!selectedPatient && (
                <div style={{ marginBottom: "1rem" }}>
                    <label htmlFor="risk-patient-select">
                        Select Example Patient:{" "}
                    </label>

                    <select
                        id="risk-patient-select"
                        value={exampleIndex}
                        onChange={(e) =>
                            setExampleIndex(Number(e.target.value))
                        }
                    >
                        {examples.map((example, index) => (
                            <option
                                key={`${example.site}-${example.patient_id}`}
                                value={index}
                            >
                                {example.patient_id} —{" "}
                                {example.developed_sepsis
                                    ? "Developed Sepsis"
                                    : "No Sepsis"}
                            </option>
                        ))}
                    </select>
                </div>
            )}

            <div style={{ marginBottom: "1rem" }}>
                <p>
                    Patient ID:{" "}
                    <strong>{patient.patient_id}</strong>
                </p>

                <p>
                    Site: <strong>{patient.site}</strong>
                </p>

                <p>
                    ICU Hour: <strong>{displayHour}</strong>
                </p>
            </div>

            {/* One row for each prediction horizon */}
            <div
                style={{
                    display: "grid",
                    gap: "0.75rem",
                    marginBottom: "1rem",
                }}
            >
                {horizons.map(({ key, label }) => {
                    // Backend results for this horizon.
                    const model = riskData?.horizons?.[key];

                    // Prefer the backend score if available.
                    const score =
                        model?.risk_score ??
                        patient[`risk_score_${key}`] ??
                        null;

                    // Never infer alert status from the score.
                    const fired = model?.fired;

                    const status =
                        fired === true
                            ? "Fired"
                            : fired === false
                              ? "Not Fired"
                              : "Awaiting backend";

                    return (
                        <div
                            key={key}
                            style={{
                                border: "1px solid #ccc",
                                borderRadius: "8px",
                                padding: "0.85rem",
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                flexWrap: "wrap",
                                gap: "0.75rem",
                            }}
                        >
                            <div>
                                <strong>
                                    {key} — {label}
                                </strong>

                                <p style={{ marginBottom: 0 }}>
                                    TFT Score:{" "}
                                    <strong>
                                        {formatScore(score)}
                                    </strong>
                                </p>
                            </div>

                            <strong>
                                {status}
                            </strong>
                        </div>
                    );
                })}
            </div>

            {/* Overall alert is controlled by backend logic */}
            <div
                style={{
                    border: "1px solid #ccc",
                    borderRadius: "8px",
                    padding: "1rem",
                    marginBottom: "1rem",
                }}
            >
                <strong>Overall Alert Level: </strong>
                <strong>{overallAlert}</strong>
            </div>

            <p className="placeholder">
                Scores are relative risk, not calibrated
                probabilities. Longer horizons can score higher
                because more hours are labeled as sepsis.
                Compare patients within the same horizon,
                not scores across horizons.
            </p>

            <p className="placeholder">
                These are experimental TFT predictions,
                not live patient monitoring. They have not
                been validated for clinical use.
            </p>

            <h3>Data Pipeline Status</h3>

            {error ? (
                <p>Pipeline statistics unavailable.</p>
            ) : !stats ? (
                <p>Loading pipeline statistics...</p>
            ) : (
                <div>
                    <p>
                        Training Patients:{" "}
                        {stats.train.patients.toLocaleString()}
                    </p>
                    <p>
                        Testing Patients:{" "}
                        {stats.test.patients.toLocaleString()}
                    </p>
                    <p>
                        Training Records:{" "}
                        {stats.train.records.toLocaleString()}
                    </p>
                    <p>
                        Testing Records:{" "}
                        {stats.test.records.toLocaleString()}
                    </p>
                    <p>
                        Patients with 25+ Training Records:{" "}
                        {stats.train.patients_25_plus.toLocaleString()}
                    </p>
                    <p>
                        Missing Values:{" "}
                        {stats.train.missing_values +
                            stats.test.missing_values}
                    </p>
                    <p>
                        Invalid Labels:{" "}
                        {stats.train.invalid_labels +
                            stats.test.invalid_labels}
                    </p>
                    <p>
                        TFT Windows (Test Subset):{" "}
                        {stats.subset.windows.toLocaleString()}
                    </p>
                </div>
            )}
        </section>
    );
}

export { RiskScore };
export default RiskScore;
