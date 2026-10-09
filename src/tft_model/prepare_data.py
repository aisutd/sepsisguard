
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


#Load the five raw patient files for initial testing
def load_raw_sample():
    frames = []
    raw_folder = PROJECT_ROOT / "data/raw"

    # Find all PSV files in the raw data folder
    for path in sorted(raw_folder.glob("*.psv")):
        patient = pd.read_csv(path, sep="|")

        # Identify the patient from the filename.
        patient["patient_id"] = path.stem
        patient["site"] = "A"

        frames.append(patient)

    if not frames:
        raise FileNotFoundError("No raw patient files found.")

    return pd.concat(frames, ignore_index=True)


def clean_raw_sample(raw_data):

    missing_cols = [
        col for col in OUTPUT_COLS
        if col not in raw_data.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    data = raw_data[OUTPUT_COLS].copy()
    data["ICULOS"] = data["ICULOS"].astype(int)

    data = data.drop_duplicates(
        subset=["site", "patient_id", "ICULOS"],
        keep="last"
    )

    # Sort patient records in chronological order
    data = data.sort_values(
        ["site", "patient_id", "ICULOS"]
    ).reset_index(drop=True)

    patient_frames = []

    for (site, patient_id), patient in data.groupby(
        ["site", "patient_id"]
    ):
        patient = patient.set_index("ICULOS")

        hours = range(
            int(patient.index.min()),
            int(patient.index.max()) + 1
        )

        patient = patient.reindex(hours)
        patient.index.name = "ICULOS"

        patient["site"] = site
        patient["patient_id"] = patient_id

        patient_frames.append(patient.reset_index())

    data = pd.concat(patient_frames, ignore_index=True)

    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    data[FEATURES] = data.groupby(
        ["site", "patient_id"]
    )[FEATURES].ffill()

    medians = data[FEATURES].median()

    all_missing = medians[medians.isna()].index.tolist()

    if all_missing:
        print("Warning: no recorded values for:", all_missing)
        print("Using zero placeholders for these test-only features.")

    medians = medians.fillna(0.0)
    data[FEATURES] = data[FEATURES].fillna(medians)

    # The label is binary (0 or 1).
    data["SepsisLabel"] = pd.to_numeric(
        data["SepsisLabel"], errors="coerce"
    )

    data["SepsisLabel"] = data.groupby(
        ["site", "patient_id"]
    )["SepsisLabel"].ffill().fillna(0).astype("float32")

    assert not data[FEATURES].isna().any().any()
    assert not data["SepsisLabel"].isna().any()
    assert data["SepsisLabel"].isin([0, 1]).all()

    return data


# Load real patient records from the raw PSV files.
train_path = PROJECT_ROOT / "data/processed/train.csv"

data = pd.read_csv(train_path, nrows=50000)

# Check how much data we have.
print("Data shape:", data.shape)

print("Number of patients:", data["patient_id"].nunique())

print("Hours per patient:")
print(data.groupby("patient_id")["ICULOS"].nunique().describe())

# Check missing values.
print("Missing values:", data[FEATURES + ["SepsisLabel"]].isna().sum().sum())

# Sort patient records in chronological order
data = data.sort_values(
    ["site", "patient_id", "ICULOS"]
).reset_index(drop=True)

# The label is binary (0 or 1).
data["SepsisLabel"] = data["SepsisLabel"].astype("float32")



# Use 24 past hours to prepare a 1-hour-ahead prediction.
max_encoder_length = 24
max_prediction_length = 1

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

print("\nPatients with enough history:",
      data[["site", "patient_id"]].drop_duplicates().shape[0])


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

