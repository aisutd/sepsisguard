from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/xgb_baseline.json")
LABEL = "SepsisLabel"

NOT_FEATURES = ["patient_id", "site", LABEL]

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

features = [c for c in train.columns if c not in NOT_FEATURES]
X_train, y_train = train[features], train[LABEL]
X_test, y_test = test[features], test[LABEL]
print(f"Training on {len(X_train):,} rows with {len(features)} features")

model = xgb.XGBClassifier(
    n_estimators=300,   
    max_depth=6,        
    learning_rate=0.1,  
    n_jobs=-1,           
)
model.fit(X_train, y_train)

scores = model.predict_proba(X_test)[:, 1]

print(f"AUROC: {roc_auc_score(y_test, scores):.3f}")
print(f"AUPRC: {average_precision_score(y_test, scores):.3f}  (random would be {y_test.mean():.3f})")

MODEL_PATH.parent.mkdir(exist_ok=True)
model.save_model(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
