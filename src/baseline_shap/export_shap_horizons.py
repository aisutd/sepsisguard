import json
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

def export_horizon_shap(model_path="models/xgb_h12.json",
                        test_data_path="data/processed/test.csv",
                        output_path="data/sample/shap_top_features_h12.json",
                        horizon=12):

    print(f"Loading model from {model_path} and test data from {test_data_path}...")
    model = xgb.Booster()
    model.load_model(model_path)
    test_df = pd.read_csv(test_data_path,nrows=5000)

    if "SepsisLabel" in test_df.columns:
        sepsis_paitents = test_df[test_df["SepsisLabel"] == 1].head(5)
        control_paitents = test_df[test_df["SepsisLabel"] == 0].head(5)
        sample_df = pd.concat([sepsis_paitents, control_paitents])
    else:
        sample_df = test_df.head(10)

    X_sample = sample_df.drop(columns=["SepsisLabel", "patient_id"], errors='ignore')
    X_sample = X_sample.select_dtypes(include=[np.number])  # Keep only numeric columns

    patient_ids = (
        sample_df["patient_id"].values
        if "patient_id" in sample_df.columns
        else [i for i in range(len(sample_df))]
    )

    print(f"Calculating SHAP values with TreeExplainer for horizon {horizon}h...")

    def model_predict(data):
        dmatrix = xgb.DMatrix(data)
        return model.predict(dmatrix)
    
    explainer = shap.Explainer(model_predict, X_sample)
    shap_values = explainer(X_sample)

    output_entries = []
    feature_names = X_sample.columns.tolist()

    # Iterate over the first 10 test patients
    for idx, pid in enumerate(patient_ids):

        raw_vals = shap_values.values
        if isinstance(raw_vals, list):
            patient_shap = raw_vals[1][idx]
        else:
            patient_shap = raw_vals[idx]

        if hasattr(patient_shap, "values"):
            patient_shap = patient_shap.values
        patient_shap = np.array(patient_shap).flatten()

        # Sort features by absolute SHAP value magnitude to find the top 5
        top_indices = np.argsort(np.abs(patient_shap))[::-1][:5]

        top_5 = [
            {
                "feature_name": feature_names[i],
                "shap_value": round(float(patient_shap[i]), 4)
            }
            for i in top_indices
        ]

        output_entries.append({
            "patient_id": str(pid),
            "horizon": horizon,
            "top_features": top_5
        })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_entries, f, indent=4)

    print(f"Successfully exported SHAP top features to {output_path}")

if __name__ == "__main__":
    export_horizon_shap(test_data_path="data/processed/test.csv",)