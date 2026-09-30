Data Pipeline Notes

Patient data in each .psv file is pipe-delimited, and each row is a single hour of data. 
Vitals Signs are columns 1 - 8, laboratory values are columns 9 - 34, demographics are columns 35 - 40, and outcome (SepsisLabel) is column 41.
Available patient data per column differs, so I calculated the missing-value percentage and organized them from highest missing values to lowest.
