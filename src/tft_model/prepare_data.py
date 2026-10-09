
#this code...

from pathlib import Path
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

#vital signs recorded for patients
VITALS = [
    "HR", "O2Sat", "Temp", "SBP",
    "MAP", "DBP", "Resp"
]

#laboratory measurements
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

FEATURES = VITALS + LABS + DEMOGRAPHICS

OUTPUT_COLS = (
    ["patient_id", "site", "ICULOS"]
    + FEATURES
    + ["SepsisLabel"]
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_data(nrows=50000):
    train_path = PROJECT_ROOT / "data/processed/train.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"File not found: {train_path}")

    data = pd.read_csv(train_path, nrows=nrows)

    return data


#checks required columns, values, chronological sorting, label conversion
# basically everything before forecasting receives the data
def validate_data(data, max_encoder_length, max_prediction_length):

    required_cols = [
        "site", "patient_id", "ICULOS", "SepsisLabel"
    ] + FEATURES

    missing_cols = [
        col for col in required_cols
        if col not in data.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if data[required_cols].isna().any().any():
        raise ValueError("Dataset contains missing values")

    # Sort patient records in chronological order
    data = data.sort_values(
        ["site", "patient_id", "ICULOS"]
    ).reset_index(drop=True)

    # The label is binary (0 or 1).
    data["SepsisLabel"] = data["SepsisLabel"].astype("float32")

    patient_lengths = data.groupby(
        ["site", "patient_id"]
    )["ICULOS"].transform("size")

    data = data[
        patient_lengths >= (
            max_encoder_length + max_prediction_length
        )
    ].copy()

    if data.empty:
        raise ValueError(
            "No patients have enough hours for a "
            "24-hour encoder and 1-hour prediction."
        )

    return data

def create_dataset(data, max_encoder_length, max_prediction_length):

    patient_dataset = TimeSeriesDataSet(
        data,

        #Identify which patient each sequence belongs to, keeps each paitent history seperate even if IDs overlap
        group_ids=["site", "patient_id"],

        #Track the order of hourly observations (hours)
        time_idx="ICULOS",

        #Value the model will eventually predict
        target="SepsisLabel",

        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        #Information that stays constant
        static_categoricals=["site"],
        static_reals=DEMOGRAPHICS,

        #hour numbers are known in advance
        time_varying_known_reals=["ICULOS"],

        #Future vitals, labs, and labels are not known
        time_varying_unknown_reals=VITALS + LABS + ["SepsisLabel"],

        #Preserve the original 0/1 target values.
        target_normalizer=None,
    )

    return patient_dataset

#only load 100k rows at a time
def inspect_full_data(filename="train.csv", chunk_size=100000):
    train_path = PROJECT_ROOT / "data/processed" / filename

    total_rows = 0
    missing_values = 0
    invalid_labels = 0
    patient_counts = {}

    for chunk in pd.read_csv(train_path, chunksize=chunk_size):
        total_rows += len(chunk)

        missing_values += chunk[
            FEATURES + ["SepsisLabel"]
        ].isna().sum().sum()

        invalid_labels += (
            ~chunk["SepsisLabel"].isin([0, 1])
        ).sum()

        counts = chunk.groupby(
            ["site", "patient_id"]
        ).size()

        for patient, count in counts.items():
            patient_counts[patient] = (
                patient_counts.get(patient, 0) + count
            )

    eligible_patients = sum(
        count >= 25 for count in patient_counts.values()
    )

    print(f"\nFull dataset inspection: {filename}")
    print("Total records:", total_rows)
    print("Total patients:", len(patient_counts))
    print("Patients with 25+ records:", eligible_patients)
    print("Missing values:", missing_values)
    print("Invalid sepsis labels:", invalid_labels)



#Load real patient records from the processed CSVs
inspect_full_data("train.csv")
inspect_full_data("test.csv")

data = load_data()



# Check how much data we have.
print("Data shape:", data.shape)

print("Number of patients:", data["patient_id"].nunique())

print("Hours per patient:")
print(data.groupby("patient_id")["ICULOS"].nunique().describe())

# Check missing values.
print("Missing values:", data[FEATURES + ["SepsisLabel"]].isna().sum().sum())



# Use 24 past hours to prepare a 1-hour-ahead prediction
max_encoder_length = 24
max_prediction_length = 1

data = validate_data(
    data,
    max_encoder_length,
    max_prediction_length
)

print("\nPatients with enough history:",
      data[["site", "patient_id"]].drop_duplicates().shape[0])


patient_dataset = create_dataset(
    data,
    max_encoder_length,
    max_prediction_length
)

print("\nTimeSeriesDataSet created successfully!")
print("Number of patient windows:", len(patient_dataset))


#dataloader
#patient windows into model-ready batches
patient_dataloader = patient_dataset.to_dataloader(
    train=False,
    batch_size=8,
    num_workers=0,
)

x, y = next(iter(patient_dataloader))

print("\nNumber of batches:", len(patient_dataloader))
print("Encoder shape:", x["encoder_cont"].shape)
print("Decoder shape:", x["decoder_cont"].shape)
print("Target shape:", y[0].shape)

print("\nPatient data pipeline completed successfully!")

