import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# Load Maritza's cleaned data
# TO DO: Update filename
train_path = os.path.join("data", "processed", "train.csv")
test_path = os.path.join("data", "processed", "test.csv")

print("Loading train data...")
train_df = pd.read_csv(train_path, nrows=20000, engine='python', on_bad_lines='skip')

print("Loading train data...")
test_df = pd.read_csv(test_path, nrows=20000, engine='python', on_bad_lines='skip')

print(f"Train data loaded! Shape: {train_df.shape}")
print(f"Test data loaded! Shape: {test_df.shape}")

# Separate features (X) and target (y)
drop_cols = ["SepsisLabel", "patient_id"]
#if "paitent_id" in train_df.columns:
 #   drop_cols.append("paitent_id")

# Convert categorical columns for XGBoost native categorical support
for col in ["site", "Gender"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype("category").cat.codes
        test_df[col] = test_df[col].astype("category").cat.codes

#X_train = train_df.drop(columns=drop_cols, errors="ignore")
X_train = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns], errors="ignore")
y_train = train_df["SepsisLabel"]

#X_test = test_df.drop(columns=drop_cols, errors="ignore")
X_test = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns], errors="ignore")
y_test = test_df["SepsisLabel"]

# Train baseline XGBoost model
print("Training XGBoost baseline model...")
model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    enable_categorical=True,
    random_state=42
)
model.fit(X_train, y_train)

# Calculate and print basic accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Model Accuracy: {accuracy:.4f}")

# Generate SHAP summary plt and save it
print("Generating SHAP summary plot...")

# Use small background sample for efficiency
background_data = X_train.sample(n=min(100, len(X_train)), random_state=42)
explainer = shap.Explainer(model.predict, background_data)

# Calculate SHAP values on test data
test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
shap_values = explainer(test_sample)

os.makedirs("docs", exist_ok=True)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_sample, show=False)
plt.savefig(os.path.join("docs", "shap_plot.png"), bbox_inches="tight")
plt.close()

print("SHAP summary plot saved successfully to docs/shap_plot.png!")

