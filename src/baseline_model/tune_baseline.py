import sys
from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(str(Path(__file__).resolve().parents[1]))  # lets us import from src/
from evaluation.evaluate import evaluate

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/xgb_baseline.json")
LABEL = "SepsisLabel"
N_TRIALS = 50   # how many settings Optuna tries (more = better results but slower)
SEED = 42      # fixed seed so results are the same every run

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
    """Train XGBoost on all training data with simple default settings plus any extras."""
    model = xgb.XGBClassifier(
        n_estimators=300,    # number of trees
        max_depth=6,         # how deep each tree can grow
        learning_rate=0.1,   # how much each new tree corrects the previous ones
        n_jobs=-1,           # use all CPU cores
        **extra_settings,
    )
    model.fit(X_train, y_train)
    return model


plain = train_xgb()
weighted = train_xgb(scale_pos_weight=spw)
evaluate(y_test, plain.predict_proba(X_test)[:, 1], name="XGBoost (no imbalance setting)")
evaluate(y_test, weighted.predict_proba(X_test)[:, 1], name=f"XGBoost (scale_pos_weight={spw:.0f})")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
fit_idx, val_idx = next(splitter.split(X_train, y_train, groups=train["patient_id"]))
X_fit, y_fit = X_train.iloc[fit_idx], y_train.iloc[fit_idx]
X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]


def objective(trial):
    """One Optuna trial: try a set of settings and return the validation AUPRC."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),       # min data needed to split a tree node
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),                # share of rows each tree sees
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),  # share of features each tree sees
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, spw, log=True),
    }
    model = xgb.XGBClassifier(
        n_estimators=1000,         # upper limit; early stopping usually ends much sooner
        early_stopping_rounds=30,  # stop adding trees once validation AUPRC stops improving
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=SEED,
        **params,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    trial.set_user_attr("n_trees", model.best_iteration + 1)  # remember how many trees worked best
    return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])


optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep the output readable
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_trees = study.best_trial.user_attrs["n_trees"]
print(f"\nBest validation AUPRC: {study.best_value:.3f}")
print(f"Best settings: {study.best_params}, n_trees={best_trees}")

final = xgb.XGBClassifier(n_estimators=best_trees, n_jobs=-1, random_state=SEED, **study.best_params)
final.fit(X_train, y_train)
evaluate(y_test, final.predict_proba(X_test)[:, 1], name="XGBoost (tuned with Optuna)")

MODEL_PATH.parent.mkdir(exist_ok=True)
final.save_model(MODEL_PATH)
print(f"Saved tuned model to {MODEL_PATH}")
