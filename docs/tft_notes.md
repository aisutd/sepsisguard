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