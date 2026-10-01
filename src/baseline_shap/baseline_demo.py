import os
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

# Load sklearn's built-in breast cancer dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# Train XGBoost classifier model on the dataset
model = xgb.XGBClassifier()
model.fit(X, y)

# Calculate SHAP values to explain how model makes predictions
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

# Make sure output folder exists before saving
os.makedirs("docs", exist_ok=True)

# Generate SHAP summary plot and save it to docs folder
shap.summary_plot(shap_values, X, show=False)
plt.savefig("docs/shap_plot.png", bbox_inches="tight")
plt.close()

print("Model trained and SHAP plot saved successfully!")