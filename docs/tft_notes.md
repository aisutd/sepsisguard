week 1 notes
 - a TimeSeriesDataSet organizes normal table data into time-series samples for a TFT to read and use
 - A sime-series model needs to see data in the order of events.
 - A Temporal Fusion Transformer (TFT) looks at sections of past data and uses it to predict a section of the future.
 It can use many features at once (For example the past 24 hours of tempreture, heart rate, and blood pressure) in order to predict

 Week 2 Notes
 - Used clean_patient.py to generate train and test data.
 - Loaded all data from train and verfied all values
 - sorted by chronological order
 - 951 patients had sufficient history for the 24-hour encoder and 1-hour target.
 - Created 2,633 DataLoader batches with a batch size of 8.
 - Vefified the tensor shapes:
   Encoder: [8, 24, 37]
   Decoder: [8, 1, 37]
   Target: [8, 1]

Week 3 Notes
- Trained TFT using 1000 paitents, spliting 80/20.
- TFT had over 12000 parameters. The model uses first 24 hours to predict the next hour.
- Did 2 epochs using 20 training batches per epoch...Training loss decreased from 0.568 to 0.323, and validation loss decreased from 0.413 to 0.237. EDIT - Increased to 3 epochs and 100 training batches per epoch. The final training loss was 0.1120 and validation loss was 0.1111, both lower than the initial run.
- AUROC of 0.7841 and AUPRC of 0.1522

- Week 4 Notes
- Scaled TFT training from 1,000 to 23,823 eligible patients using my Apple M5 GPU, with a fixed 80/20 patient-level split (19,059 training, 4,764 validation).
- Trained both 6-hour and 12-hour sepsis prediction models using the same patients, 24-hour history, batch size of 32, and 3 epochs. Only the prediction target changed.
- Achieved AUROC 0.817 / AUPRC 0.123 (6h)** and AUROC 0.819 / AUPRC 0.184 (12h) using Harish's evaluation script.
- Generated risk scores for 127,064 matching patient-hours across 5,972 test patients. The TFT could not score 2,095 patients due to insufficient history.
- Added model checkpointing, removed potential label leakage, and updated the dashboard to display real sepsis-positive and non-sepsis patient examples. Both models showed signs of overfitting after the first epoch, so the best checkpoints were used.