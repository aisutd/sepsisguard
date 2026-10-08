SHAP Baseline Nptes

Top Features: 
- Will add once model runs

Observations:
- 

Couldn't find ShapChat().jsv

function ShapChart(){
    //Anikait's work
    const topFeatures = [
        { name: "Feature 1 (Pending)", value: 0.0},
        { name: "Feature 2 (Pending)", value: 0.0},
        { name: "Feature 3 (Pending)", value: 0.0},
        { name: "Feature 3 (Pending)", value: 0.0}
    ];

    return(
        <section className="panel">
            <h2>SHAP Feature Importance (Baseline)</h2>
            <u1 className="shap-list>
                {topFeatures.map((feat, index) => (
                    <li key={index}>
                        <span>{feat.name}</span>: <span>{feat.value}</span>
                    </li>
                ))}
            </section>
    );
}

export default ShapChart;