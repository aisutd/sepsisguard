from pathlib import Path
import pandas as pd

# Folder where patient files (.psv) are located
RAW_DATA_DIR = Path("data/raw")

# Load all patient files and combine into one DataFram
def load_patients():

    #Find all .psv files in data/raw and its subfolder, put them in alphabetical order
    patient_files = sorted(RAW_DATA_DIR.rglob("*.psv"))

    # Validate file finding
    if not patient_files:
        raise FileNotFoundError(
            f"No .psv files found in {RAW_DATA_DIR}"
        )

    # List for DataFrame of each patient
    dataframes = []

    #Read each patient file one at a time
    for file in patient_files:

        #Tell pandas to use "|" as separator because it's used in the datasets to separate columns
        df = pd.read_csv(file, sep="|")

        # Add the patient's filename as their ID to easily identify which rows belong to which patient
        df["patient_id"] = file.stem

        # Add the individual patient DataFrame to list
        dataframes.append(df)

    # Combine all DataFrames into one large DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

if __name__ == "__main__":

    # Load all patient files into one DtaFrame
    patients = load_patients()

    #Print total shape of the Combined DataFrmae (rows, columns)
    print("\nTotal shape:") 
    print(patients.shape)

    # Calculate the percentage of missing values in each column
    # by finding the missing valu (isna()), calculating the percentage of those values, then multiplying by 100 
    #missing_percent = patients.isna().mean() * 100 #Not organized
    missing_percent = (patients.isna().mean() * 100).sort_values(ascending=False)

    print("\nMissing-value percentage per column (hi --> lo):\n")

    # Find longest column name and pad every name to same width 
    name_width = max(len(column) for column in missing_percent.index)

    
    #Print each column and its percentage of missing values
    for column, percent in missing_percent.items():
        print(f"{column:<{name_width}}: {percent:>8.2f}%")



    