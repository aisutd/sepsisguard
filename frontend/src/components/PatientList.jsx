import { useEffect, useState } from "react"; //pulls two tools from react library

// Shows real patients from our cleaned data. The file is
// data/processed/sample_patients.csv (made by clean_patients.py), copied into
// frontend/public/ so the browser can load it. Risk is a placeholder until
// Arnav's model gives us real scores.

function PatientList()
{
    const [patients, setPatients] = useState([]); //rows from the csv
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch("/sample_patients.csv") //files in frontend/public/ are served at "/"
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`status ${response.status}`);
                }
                return response.text(); //the csv as one long string
            })
            .then((text) => {
                // First line is the header (column names), the rest are patients.
                // Splitting on "," is safe here: the file only has ids and numbers.
                const [header, ...lines] = text.trim().split("\n");
                const columns = header.trim().split(",");
                const rows = lines.map((line) => {
                    const values = line.trim().split(",");
                    // e.g. { patient_id: "p000006", site: "A", ICULOS: "19", ... }
                    return Object.fromEntries(columns.map((col, i) => [col, values[i]]));
                });
                setPatients(rows.slice(0, 20)); //show the first 20 patients
            })
            .catch((err) => setError(err.message));
    }, []);

    return(
        <section className="panel">
            <h2>Patients</h2>

            {error && <p className="placeholder">Couldn't load patients: {error}</p>}

            <table className="patient-table">
                <thead>
                    <tr>
                        <th>Patient</th>
                        <th>Site</th>
                        <th>ICU hrs</th>
                        <th>Risk</th>
                    </tr>
                </thead>
                <tbody>
                    {patients.map((p) => (
                        <tr key={p.patient_id}>
                            <td>{p.patient_id}</td>
                            <td>{p.site}</td>
                            <td>{p.ICULOS}</td>
                            {/* placeholder: real risk score comes from Arnav's model later */}
                            <td className="placeholder">Pending</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </section>
    );
}

export default PatientList;
