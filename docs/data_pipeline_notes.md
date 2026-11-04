Data Pipeline Notes

Patient data in each .psv file is pipe-delimited, and each row is a single hour of data. 
Vitals Signs are columns 1 - 8, laboratory values are columns 9 - 34, demographics are columns 35 - 40, and outcome (SepsisLabel) is column 41.
Available patient data per column differs, so I calculated the missing-value percentage and organized them from highest missing values to lowest.

Week 2
I combined training sets A and B into 40,336 patients (1,552,210 hourly rows) and kept 39 columns, dropping Bilirubin_direct, EtCO2, Fibrinogen, and 
TroponinI because each is missing in over 95% of rows and measured for under 30% of patients (EtCO2 is also 100% missing at Site A). For missing values
I forward-filled within each patient, since a measurement stays current until it is re-measured and this never uses future data, then filled anything still 
empty with the training-set median so no test data leaks into training. Site A and Site B look similar after cleaning, but Site A has a higher sepsis rate 
(8.8% vs 5.7%) and different measuring habits (e.g. DBP is missing 48% of the time at A vs 14% at B), so model performance should be checked per site later. 
Only 7.27% of patients (2,932) are ever sepsis-positive and only 1.80% of hourly rows are labeled. The data is split 80/20 by patient, keeping the 7.27% rate 
in both sets, into data/processed/train.csv and test.csv 

Week 4 Notes
I checked the PhysioNet 2019 Challenge page: for sepsis patients SepsisLabel is 1 when t ≥ t_sepsis − 6 (6 hours before clinical onset, which the challenge defines using the Sepsis-3 criteria) and 0 before, and non-sepsis patients are always 0, which matches the issue. In make_horizon_labels.py I recovered each septic patient's onset as their first hour labeled 1 plus 6, then built label_6h, label_12h and label_24h, which are 1 from (onset − H) hours onward; label_6h matches SepsisLabel exactly (0 mismatches in train and test). All 2,932 septic patients (2,346 train, 586 test) are positive at every horizon, but the share of hourly rows labeled 1 grows from 1.8% (H = 6) to 2.7% (H = 12) and 4.2% (H = 24). The number of septic patients already labeled 1 at their first hour also grows, from 426 (15%) at H = 6 to 778 (27%) at H = 12 and 1,236 (42%) at H = 24, so at longer horizons many patients give the model no "before" period to learn from, and their true onset may be even earlier than computed. In make_trend_features.py I added 10 trend columns, the change over the last 6 and 12 hours (value now minus value then, within the same patient and never using future hours) for HR, Resp, Temp, MAP and O2Sat, set to 0 for the first hours of each stay. The patient split is unchanged and the final files are data/processed/train_features.csvand test_features.csv.



 