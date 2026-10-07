Data Pipeline Notes

Patient data in each .psv file is pipe-delimited, and each row is a single hour of data. 
Vitals Signs are columns 1 - 8, laboratory values are columns 9 - 34, demographics are columns 35 - 40, and outcome (SepsisLabel) is column 41.
Available patient data per column differs, so I calculated the missing-value percentage and organized them from highest missing values to lowest.

Week 2
I combined training sets A and B into 40,336 patients (1,552,210 hourly rows) and kept 39 columns, dropping Bilirubin_direct, EtCO2, Fibrinogen, and TroponinI because each is missing in over 95% of rows and measured for under 30% of patients (EtCO2 is also 100% missing at Site A). For missing values I forward-filled within each patient, since a measurement stays current until it is re-measured and this never uses future data, then filled anything still empty with the training-set median so no test data leaks into training. Site A and Site B look similar after cleaning, but Site A has a higher sepsis rate (8.8% vs 5.7%) and different measuring habits (e.g. DBP is missing 48% of the time at A vs 14% at B), so model performance should be checked per site later. Only 7.27% of patients (2,932) are ever sepsis-positive and only 1.80% of hourly rows are labeled. The data is split 80/20 by patient, keeping the 7.27% rate in both sets, into data/processed/train.csv and test.csv 

