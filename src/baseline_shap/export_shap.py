import json
import os
import xgboost as xgb
import pandas as pd
import numpy as np
import shap

def main():
    # Define paths for model, test data, and output JSON
    model_path = "models/xgb_baseline.json" # Harish's model
    test_data_path = "data/processed/test.csv" 
    output_path = "data/sample/shap_top_features.json"

    # Load the trained model and test data
    print("Loading model and test data...")
    model = xgb.Booster()
    model.load_model(model_path)
    test_df = pd.read_csv(test_data_path)

    # Select about 10 patients (mixing sepsis and non-sepsis) for SHAP analysis
    if "SepsisLabel" in test_df.columns:
        sepsis_paitents = test_df[test_df["SepsisLabel"] == 1].head(5)
        control_paitents = test_df[test_df["SepsisLabel"] == 0].head(5)
        sample_df = pd.concat([sepsis_paitents, control_paitents])
    else:
        sample_df = test_df.head(10)

    # Drop target or patient_id columns from feature matrix if present
    X_sample = sample_df.drop(columns=["SepsisLabel", "patient_id"], errors="ignore")
    X_sample = X_sample.select_dtypes(include=[np.number])
      # Ensure only numeric features are used
    patient_ids = sample_df["patient_id"].values if "patient_id" in sample_df.columns else [f"patient_{i}" for i in range(len(sample_df))]

    # Calculate SHAP values for the selected patients using the trained model
    print("Calculating SHAP values...")

    def model_predict(data):
        dmatrix = xgb.DMatrix(data)
        return model.predict(dmatrix)

    explainer = shap.Explainer(model_predict, X_sample)
    shap_values = explainer(X_sample)

    # Extract top 5 features by absolute SHAP values for each patient
    output_entries = []

    for idx, pid in enumerate(patient_ids):
        # Get SHAP values for the current patient
        paitent_shap_values = shap_values.values[idx]
        feature_names = X_sample.columns.tolist()

        # Combine featue names, values and absolute SHAP values for sorting
        feature_shap_pairs = []
        for feature, s_val in zip(feature_names, paitent_shap_values):
            feature_shap_pairs.append({
                "feature": feature,
                "shap_value": float(s_val),
                "abs_shap": abs(s_val)
            })

        # Sort by absolute SHAP values in descending order and keep top 5 features
        feature_shap_pairs.sort(key=lambda x: x["abs_shap"], reverse=True)
        top_5 = [{"feature": item["feature"], "shap_value": round(item["shap_value"], 4)} for item in feature_shap_pairs[:5]]

        output_entries.append({
            "patient_id": str(pid),
            "top_features": top_5
        })

    # Check output directory exists and save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_entries, f, indent=2)

    print(f"Successfully exported SHAP top features for {len(output_entries)} patients to {output_path}")

if __name__ == "__main__":
    main()
