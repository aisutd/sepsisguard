
import { useEffect, useState } from "react";

function RiskScore() {
    const [stats, setStats] = useState(null);
    const [error, setError] = useState(false);

    useEffect(() => {
        fetch(`${import.meta.env.BASE_URL}pipeline-status.json`)
            .then(response => {
                if (!response.ok) {
                    throw new Error("Could not load pipeline status");
                }
                return response.json();
            })
            .then(data => setStats(data))
            .catch(() => setError(true));
    }, []);

    return (
        <section className="panel">
            <h2>Risk Score</h2>

            <p className="placeholder">
                Risk score coming soon...
            </p>

            <h3>First TFT Prediction</h3>

            <p>
                Patient ID: <strong>p000619</strong>
            </p>

            <p>
                Experimental sepsis prediction score: <strong>1.25%</strong>
            </p>

            <p className="placeholder">
                This is a real prediction from the updated Week 3 TFT training run.
                It is a static example, not a live prediction.
                The model has not been validated for clinical use.
            </p>

            <h3>Data Pipeline Status</h3>

            {error ? (
                <p>Pipeline statistics unavailable.</p>
            ) : !stats ? (
                <p>Loading pipeline statistics...</p>
            ) : (
                <div>
                    <p>Training Patients: {stats.train.patients.toLocaleString()}</p>
                    <p>Testing Patients: {stats.test.patients.toLocaleString()}</p>
                    <p>Training Records: {stats.train.records.toLocaleString()}</p>
                    <p>Testing Records: {stats.test.records.toLocaleString()}</p>
                    <p>Patients with 25+ Training Records: {stats.train.patients_25_plus.toLocaleString()}</p>
                    <p>Missing Values: {stats.train.missing_values + stats.test.missing_values}</p>
                    <p>Invalid Labels: {stats.train.invalid_labels + stats.test.invalid_labels}</p>
                    <p>TFT Windows (Test Subset): {stats.subset.windows.toLocaleString()}</p>

                    <p className="placeholder">
                        Dataset preparation complete. Model predictions are not available yet.
                    </p>
                </div>
            )}
        </section>
    );
}

export default RiskScore;
