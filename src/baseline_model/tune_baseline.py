import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[1]))  # lets us import from src/
from evaluation.evaluate import evaluate

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

spw = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {spw:.1f}")


def train_xgb(**extra_settings):
    """Train XGBoost with our base settings, plus any extra settings passed in."""
    model = xgb.XGBClassifier(
        n_estimators=300,   
        max_depth=6,        
        learning_rate=0.1,  
        n_jobs=-1,           
        **extra_settings,
    )
    model.fit(X_train, y_train)
    return model

plain = train_xgb()
weighted = train_xgb(scale_pos_weight=spw)

evaluate(y_test, plain.predict_proba(X_test)[:, 1], name="XGBoost (no imbalance setting)")
evaluate(y_test, weighted.predict_proba(X_test)[:, 1], name=f"XGBoost (scale_pos_weight={spw:.0f})")

MODEL_PATH.parent.mkdir(exist_ok=True)
plain.save_model(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
