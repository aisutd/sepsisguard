// Shows real patients from our sample file data/sample/patients_sample.json
// (made by src/data_pipeline/export_sample.py). 


// Each patient's latest vitals from their last hour in the ICU,
// whether they developed sepsis, and how complete their data is.
const patients = [
    { patient_id: "p113489", site: "B", hours_in_icu: 52, heart_rate: 101.5, resp_rate: 15.5, temperature: 37.4, developed_sepsis: true, confidence: "high" },
    { patient_id: "p015488", site: "A", hours_in_icu: 11, heart_rate: 112.5, resp_rate: 33.75, temperature: 38.11, developed_sepsis: true, confidence: "high" },
    { patient_id: "p009912", site: "A", hours_in_icu: 15, heart_rate: 119.0, resp_rate: 32.0, temperature: 35.94, developed_sepsis: true, confidence: "high" },
    { patient_id: "p015617", site: "A", hours_in_icu: 18, heart_rate: 82.0, resp_rate: 16.0, temperature: 36.06, developed_sepsis: false, confidence: "high" },
    { patient_id: "p113981", site: "B", hours_in_icu: 37, heart_rate: 62.0, resp_rate: 16.0, temperature: 36.7, developed_sepsis: false, confidence: "high" },
    { patient_id: "p116438", site: "B", hours_in_icu: 24, heart_rate: 55.0, resp_rate: 16.0, temperature: 36.4, developed_sepsis: false, confidence: "high" },
];

function PatientList()
{
    return(
        <section className="panel">
            <h2>Patients</h2>

            {/* The sidebar is narrow (280px), so only the key columns are shown.
                Resp rate, ICU hours and confidence are still in the patients list above
                for the detail panels to use later. */}
            <div style={{ overflowX: "auto" }}>  {/* scroll sideways instead of spilling out */}
                <table className="patient-table">
                    <thead>
                        <tr>
                            <th>Patient</th>
                            <th>HR</th>
                            <th>Temp</th>
                            <th>Sepsis</th>
                            <th>Risk</th>
                        </tr>
                    </thead>
                    <tbody>
                        {/* one table row per patient */}
                        {patients.map((p) => (
                            <tr key={p.patient_id}>
                                <td>{p.patient_id}</td>
                                <td>{p.heart_rate}</td>
                                <td>{p.temperature}</td>
                                {/* true/false shown as Yes/No */}
                                <td>{p.developed_sepsis ? "Yes" : "No"}</td>
                                {/* placeholder: real risk score comes from Arnav's model later */}
                                <td className="placeholder">Pending</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

        </section>
    );
}

export default PatientList;
