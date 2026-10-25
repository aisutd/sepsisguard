

import { useEffect, useState } from "react";

// Real patient examples from the Week 4 TFT test predictions.
// These match data/sample/predictions_sample.json.
const examples = [
    {
        site: "A",
        patient_id: "p007837",
        ICULOS: 116,
        developed_sepsis: true,
        risk_score_6h: 0.2125,
        risk_score_12h: 0.228915,
    },
    {
        site: "B",
        patient_id: "p102981",
        ICULOS: 33,
        developed_sepsis: false,
        risk_score_6h: 0.003162,
        risk_score_12h: 0.004014,
    },
];

function RiskScore() {
    const [stats, setStats] = useState(null);
    const [error, setError] = useState(false);

    // Load the existing data pipeline statistics.
    useEffect(() => {
        fetch(`${import.meta.env.BASE_URL}pipeline-status.json`)
            .then((response) => {
                if (!response.ok) {
                    throw new Error("Could not load pipeline status");
                }
                return response.json();
            })
            .then((data) => setStats(data))
            .catch(() => setError(true));
    }, []);

    return (
        <section className="panel">
            <h2>Risk Score</h2>

            <h3>Real TFT Prediction Examples</h3>

            <div
                style={{
                    display: "grid",
                    gridTemplateColumns:
                        "repeat(auto-fit, minmax(230px, 1fr))",
                    gap: "1rem",
                }}
            >
                {examples.map((patient) => (
                    <div
                        key={`${patient.site}-${patient.patient_id}`}
                        style={{
                            border: "1px solid #ccc",
                            borderRadius: "8px",
                            padding: "1rem",
                        }}
                    >
                        <h4>
                            {patient.developed_sepsis
                                ? "Patient Developed Sepsis"
                                : "Patient Did Not Develop Sepsis"}
                        </h4>

                        <p>
                            Patient ID:{" "}
                            <strong>{patient.patient_id}</strong>
                        </p>

                        <p>
                            Site: <strong>{patient.site}</strong>
                        </p>

                        <p>
                            ICU Hour:{" "}
                            <strong>{patient.ICULOS}</strong>
                        </p>

                        <p>
                            6h TFT Score:{" "}
                            <strong>
                                {(patient.risk_score_6h * 100).toFixed(2)}%
                            </strong>
                        </p>

                        <p>
                            12h TFT Score:{" "}
                            <strong>
                                {(patient.risk_score_12h * 100).toFixed(2)}%
                            </strong>
                        </p>

                        <p>
                            Actual Outcome:{" "}
                            <strong>
                                {patient.developed_sepsis
                                    ? "Developed Sepsis"
                                    : "No Sepsis"}
                            </strong>
                        </p>
                    </div>
                ))}
            </div>

            <p className="placeholder">
                These are real predictions from the Week 4 TFT
                models, not live patient monitoring. The scores
                are experimental and have not been validated
                for clinical use.
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
