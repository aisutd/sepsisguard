# This code will
# Load small subsets of paitents from train.csv
# create training and validation dataloaders
# train the TFT
# generate a prediction

import json
from pathlib import Path
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import torch
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
from sklearn.metrics import roc_auc_score, average_precision_score

# Vital signs recorded for patients
VITALS = [
    "HR", "O2Sat", "Temp", "SBP",
    "MAP", "DBP", "Resp"
]

# Laboratory measurements
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
    "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine",
    "Glucose", "Lactate", "Magnesium",
    "Phosphate", "Potassium", "Bilirubin_total",
    "Hct", "Hgb", "PTT", "WBC", "Platelets"
]

DEMOGRAPHICS = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime"
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data/processed/train.csv"
SPLIT_PATH = Path.home() / "sepsisguard_runs/week4/patient_split.csv"


# Get cleaned data from paitents with 24+ hours of it, then select 1000 paitents at random
def load_training_sample(n_patients=1000):

    # Load Maritza's cleaned training data
    data = pd.read_csv(TRAIN_PATH)

    # Count recorded hours and check for sepsis labels per patient
    patient_info = data.groupby(
        ["site", "patient_id"], as_index=False
    ).agg(
        hours=("ICULOS", "nunique"),
        ever_septic=("SepsisLabel", "max")
    )

    # Keep patients with enough history for a 24-hour encoder
    eligible = patient_info[patient_info["hours"] >= 25]

    if eligible.empty:
        raise ValueError("No patients have enough history.")

    # Randomly select patients, keeping their histories
    selected = eligible.sample(
        n=min(n_patients, len(eligible)),
        random_state=42 # Keeps same random sample every time script is ran
    )

    data = data.merge(
        selected[["site", "patient_id"]],
        on=["site", "patient_id"],
        how="inner"
    )

    data = data.sort_values(
        ["site", "patient_id", "ICULOS"]
    ).reset_index(drop=True)

    return data

# Split paitents into 80% for training and 20% for validation
# However, blindly splitting might cause a set to have no positive sepsis paitents
# Use of a stratified split identifies positive paitents and distributes them evenly across both sets
def split_train_validation(data):

    # Identify each patient and whether they developed sepsis
    patient_info = data.groupby(
        ["site", "patient_id"], as_index=False
    ).agg(
        ever_septic=("SepsisLabel", "max")
    )

    validation_groups = []

    # Keep septic and non-septic patients in both groups
    for label, group in patient_info.groupby("ever_septic"):

        if len(group) < 2:
            raise ValueError(
                f"Not enough patients in class {label}. "
                "Increase the training sample size."
            )

        n_validation = max(1, round(len(group) * 0.2))

        selected = group.sample(
            n=n_validation,
            random_state=42
        )

        validation_groups.append(selected)

    validation_ids = pd.concat(
        validation_groups,
        ignore_index=True
    )[["site", "patient_id"]]

    # Mark which patient records belong to validation
    data = data.merge(
        validation_ids.assign(is_validation=True),
        on=["site", "patient_id"],
        how="left"
    )

    # Separate training and validation records
    train_data = data[
        data["is_validation"].isna()
    ].drop(columns=["is_validation"]).copy()

    val_data = data[
        data["is_validation"] == True
    ].drop(columns=["is_validation"]).copy()

    return train_data, val_data


# Reuse the same patients for the 6-hour and 12-hour models.
def load_fixed_split(data):

    # Load the patient assignments saved earlier.
    patient_splits = pd.read_csv(SPLIT_PATH)

    # Make sure each patient has only one assignment.
    if patient_splits.duplicated(["site", "patient_id"]).any():
        raise ValueError("Duplicate patient IDs in saved split.")

    # Attach each patient's assigned group to their hourly records.
    data = data.merge(
        patient_splits,
        on=["site", "patient_id"],
        how="left",
        validate="many_to_one",
    )

    # Stop if any patient is missing from the saved split.
    if data["split"].isna().any():
        raise ValueError("Some patients are missing from the saved split.")

    if not data["split"].isin(["train", "validation"]).all():
        raise ValueError("Invalid split assignment.")

    # Separate training and validation patient records.
    train_data = data[data["split"] == "train"].drop(
        columns=["split"]
    ).copy()

    val_data = data[data["split"] == "validation"].drop(
        columns=["split"]
    ).copy()

    return train_data, val_data


#Same function as seen in prepare_data.py but now using binary classification
def create_dataloaders(train_data, val_data, batch_size=32):

    train_data = train_data.copy()
    val_data = val_data.copy()

    # Convert the binary sepsis labels into categories for classification
    for data in [train_data, val_data]:
        data["SepsisClass"] = data["SepsisLabel"].astype(int).astype(str)

        data.sort_values(
            ["site", "patient_id", "ICULOS"],
            inplace=True
        )

    # Build the training dataset using 24 hours of history
    train_dataset = TimeSeriesDataSet(
        train_data,

        group_ids=["site", "patient_id"],
        time_idx="ICULOS",
        target="SepsisClass",

        max_encoder_length=24,
        min_encoder_length=24,
        max_prediction_length=1,
        min_prediction_length=1,

        static_categoricals=["site"],
        static_reals=DEMOGRAPHICS,

        time_varying_known_reals=["ICULOS"],
        time_varying_unknown_reals=VITALS + LABS,
        time_varying_unknown_categoricals=["SepsisClass"],

        target_normalizer=NaNLabelEncoder(),
    )

    # Reuse the training encoders and scalers for validation
    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        val_data,
        predict=False,
        stop_randomization=True
    )

    # Create batches for model training and validation
    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0
    )

    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0
    )

    return train_dataset, val_dataset, train_loader, val_loader

#Initialize and run TFT
def train_model(train_dataset, train_loader, val_loader):

    # Make the training run reproducible
    seed_everything(42, workers=True)

    # Initialize the Temporal Fusion Transformer
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.001,
        hidden_size=8,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=4,
        output_size=2,
        loss=CrossEntropy(),
        logging_metrics=torch.nn.ModuleList([]),
        log_interval=-1,
    )

    print("\nTFT parameters:", tft.size())

    # Configure a small training run
    trainer = Trainer(
        max_epochs=1,
        accelerator="mps",
        devices=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        gradient_clip_val=0.1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )

    # Train the model and evaluate on validation batches
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Display the losses recorded during training
    print("\nTraining results:")

    for name, value in trainer.callback_metrics.items():
        if "loss" in name:
            print(f"{name}: {value.item():.4f}")

    return tft


def generate_predictions(tft, train_dataset):

    test_path = PROJECT_ROOT / "data/processed/test.csv"
    test_data = pd.read_csv(test_path)

    # Find test patients with at least 25 recorded hours
    patient_info = test_data.groupby(
        ["site", "patient_id"], as_index=False
    ).agg(
        hours=("ICULOS", "nunique"),
        ever_septic=("SepsisLabel", "max")
    )

    eligible = patient_info[patient_info["hours"] >= 25]

    # Select 10 positive and 10 negative test patients
    positive = eligible[eligible["ever_septic"] == 1].sample(
        n=10, random_state=42
    )
    negative = eligible[eligible["ever_septic"] == 0].sample(
        n=10, random_state=42
    )

    selected = pd.concat([positive, negative])

    test_data = test_data.merge(
        selected[["site", "patient_id"]],
        on=["site", "patient_id"],
        how="inner"
    )

    test_data["SepsisClass"] = (
        test_data["SepsisLabel"].astype(int).astype(str)
    )

    test_data = test_data.sort_values(
        ["site", "patient_id", "ICULOS"]
    ).reset_index(drop=True)

    # Reuse training encoders and select one final window per patient
    test_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        test_data,
        predict=True,
        stop_randomization=True
    )

    # Obtain raw model logits and the matching patient IDs
    predictions = tft.predict(
        test_dataset,
        mode=("raw", "prediction"),
        return_index=True,
        trainer_kwargs={"accelerator": "mps", "devices": 1}
    )

    # Convert class logits into probabilities
    positive_class = train_dataset.target_normalizer.classes_["1"]
    probabilities = torch.softmax(
        predictions.output, dim=-1
    )[:, 0, positive_class].cpu().tolist()

    results = [
        {
            "patient_id": str(patient_id),
            "risk_score": round(float(score), 4)
        }
        for patient_id, score in zip(
            predictions.index["patient_id"],
            probabilities
        )
    ]

    # Save the actual model-generated predictions
    output_path = (
        PROJECT_ROOT / "data/sample/predictions_sample.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\nPredictions generated:", len(results))
    print("Predictions saved to:", output_path)
    print("Sample predictions:", results[:3])

    return results

# AUROC and AUPRC generation
def evaluate_model(tft, train_dataset):

    test_path = PROJECT_ROOT / "data/processed/test.csv"
    test_data = pd.read_csv(test_path)

    # Find patients with enough history
    patient_info = test_data.groupby(
        ["site", "patient_id"], as_index=False
    )["ICULOS"].nunique()

    eligible = patient_info[
        patient_info["ICULOS"] >= 25
    ]

    # Randomly select 500 test patients and keep all their hours
    selected = eligible.sample(
        n=min(500, len(eligible)),
        random_state=42
    )[["site", "patient_id"]]

    test_data = test_data.merge(
        selected,
        on=["site", "patient_id"],
        how="inner"
    )

    test_data["SepsisClass"] = (
        test_data["SepsisLabel"].astype(int).astype(str)
    )

    test_data = test_data.sort_values(
        ["site", "patient_id", "ICULOS"]
    ).reset_index(drop=True)

    # Reuse the training dataset's encoders and scalers
    test_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        test_data,
        predict=False,
        stop_randomization=True
    )

    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=0
    )

    # Get model predictions and matching real labels
    predictions = tft.predict(
        test_loader,
        mode=("raw", "prediction"),
        return_y=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "logger": False
        }
    )

    # Convert model outputs into probabilities for sepsis class 1
    positive_class = train_dataset.target_normalizer.classes_["1"]

    y_scores = torch.softmax(
        predictions.output,
        dim=-1
    )[..., positive_class].reshape(-1).cpu().numpy()

    # Get the actual label for each predicted hour
    y_true = predictions.y[0].reshape(-1).cpu().numpy()

    if len(y_true) != len(y_scores):
        raise ValueError("Predictions and labels do not match.")

    if len(set(y_true.tolist())) < 2:
        raise ValueError(
            "Need both positive and negative labels to calculate AUROC."
        )

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    print("\nTest Evaluation:")
    print("Patients sampled:", len(selected))
    print("Windows evaluated:", len(y_true))
    print("Positive windows:", int(y_true.sum()))
    print("Positive rate:", round(float(y_true.mean()), 4))
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC (Average Precision): {auprc:.4f}")

    return auroc, auprc

# What runs when script is ran in order to check everything is correct
if __name__ == "__main__":

    data = load_training_sample(n_paitents=23823)

    train_data, val_data = load_fixed_split(data)

    print("Total patients:", data.groupby(
        ["site", "patient_id"]
    ).ngroups)

    print("Training patients:", train_data.groupby(
        ["site", "patient_id"]
    ).ngroups)

    print("Validation patients:", val_data.groupby(
        ["site", "patient_id"]
    ).ngroups)

    print("Sepsis-positive training patients:", (
        train_data.groupby(
            ["site", "patient_id"]
        )["SepsisLabel"].max() == 1
    ).sum())

    print("Sepsis-positive validation patients:", (
        val_data.groupby(
            ["site", "patient_id"]
        )["SepsisLabel"].max() == 1
    ).sum())

    train_ids = set(zip(
        train_data["site"],
        train_data["patient_id"]
    ))

    val_ids = set(zip(
        val_data["site"],
        val_data["patient_id"]
    ))

    print("Patients shared between groups:",
          len(train_ids & val_ids))

    # Dataloader check
    train_dataset, val_dataset, train_loader, val_loader = (
        create_dataloaders(train_data, val_data)
    )

    print("\nTraining windows:", len(train_dataset))
    print("Validation windows:", len(val_dataset))

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    x, y = next(iter(train_loader))

    print("\nEncoder shape:", x["encoder_cont"].shape)
    print("Decoder shape:", x["decoder_cont"].shape)
    print("Target shape:", y[0].shape)
    print("Target data type:", y[0].dtype)

    # Check tft
    tft = train_model(
    train_dataset,
    train_loader,
    val_loader
    )

    # Check Predictions 
    predictions = generate_predictions(tft, train_dataset)

    auroc, auprc = evaluate_model(tft, train_dataset)



