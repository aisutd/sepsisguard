function ShapChart(){
    // Define source model and paitent ID variables
    const sourceModel = "XGBoost baseline";
    const paitentId = "p000011"

    // Baseline importance scores derived from SHAP model output
    const topFeatures = [
        { name: "ICULOS", value: -0.0333},
        { name: "Resp", value: -0.0159},
        { name: "HR", value: -0.0084},
        { name: "SBP", value: -0.0047},
        { name: "Lactate", value: 0.0039}
    ];

    return(
        // Main panel container wrapper for styling
       <section className="panel">
            {/* Section heading text */}
            <h2>SHAP Feature Importance (Baseline)</h2>
            
            {/* Unordered list container for the features */}
            <ul className="shap-list">
                {/* Dynamically map over the topFeatures array to render each item */}
                {topFeatures.map((feat, index) => (
                    /* Individual listitem with a unique key required by React */
                    <li key={index}>
                        <span>{feat.name}</span>: <span>{feat.value}</span>
                    </li>
                ))}
            </ul>

        <footer className="shap-footer">
            <p>Reasons come from {sourceModel}</p>
        </footer>
    </section>
    );
}

export default ShapChart;
