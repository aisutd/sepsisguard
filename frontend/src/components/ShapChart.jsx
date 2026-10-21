//Neelimas work goes here
function ShapChart(){
    // Baseline importance scores derived from SHAP model output
    const topFeatures = [
        { name: "ICULOS", value: -0.0174},
        { name: "Temp", value: -0.0096},
        { name: "Resp", value: -0.0069},
        { name: "HR", value: -0.0054},
        { name: "Lactate", value: 0.0029}
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
