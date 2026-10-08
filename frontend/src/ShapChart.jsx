function ShapChart(){
    //Anikait's work
    // Baseline importance scores derived from SHAP model output
    const topFeatures = [
        { name: "ICULOS", value: 0.19},
        { name: "Magnesium", value: 0.14},
        { name: "WBC", value: 0.12},
        { name: "Glucose", value: 0.09}
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
        </section>
    );
}

export default ShapChart;