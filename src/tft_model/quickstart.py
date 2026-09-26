#This co
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from pytorch_forecasting.data.examples import get_stallion_data #quickstart example 

#data contains beverage related sales for different products
data = get_stallion_data()
#print(data.head())

#give every year-month its own time index
data["time_idx"] = (
    data["date"].dt.year * 12
    + data["date"].dt.month
)

#start months/index at 0
data["time_idx"] -= data["time_idx"].min()
#print(data[["date", "time_idx"]].head())

#use past 24 months to predict next 6 months
max_encoder_length = 24
max_prediction_length = 6
#80/20 split
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],

    #order
    time_idx="time_idx",

    #want to predict volume
    target="volume",

    # Each agency + SKU combination represents its own time series.
    group_ids=["agency", "sku"],

    # How much past/future history the model sees.
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,

    #info that doesnt chage over time, only acts as identefiers
    static_categoricals=["agency", "sku"],

    #info that changes over time but model knows future values(eg. indexes/calender info)
    time_varying_known_reals=[
        "time_idx",
    ],

    #info that changes over time but model DOES NOT know future values
    time_varying_unknown_reals=[
        "volume",
    ],
)
#checks
#print(training)
#print("Number of training samples:", len(training)) #training windows created

#x, y = training[0]
#print(x.keys())
#print(y)



#Dataloader- take model-ready windows and put them into batches so that model trains one batch at a time
batch_size = 64

train_dataloader = training.to_dataloader(
    train=True,
    batch_size=batch_size,
    num_workers=0,
)
#print("Number of batches:", len(train_dataloader))


#validation dataset
validation = TimeSeriesDataSet.from_dataset(
    training,
    data,
    min_prediction_idx=training_cutoff + 1, #start predictions after trianing index ends
    stop_randomization=True, #validate on same window as trained
)

#dataLoader for validation.
val_dataloader = validation.to_dataloader(
    train=False, #false in order to evaluate, not train
    batch_size=batch_size,
    num_workers=0,
)

print("Validation samples:", len(validation))
print("Validation batches:", len(val_dataloader))


#create TFT
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03, #weight change: might change later
    hidden_size=16,
    attention_head_size=1, #how far back into the months is the data useful
    dropout=0.1, #prevent overfitting
    hidden_continuous_size=8, #space given to model to process each feature
    output_size=7, #give range of possible outcomes instead of just 1 
    loss=QuantileLoss(),
)
print(f"Number of model parameters: {tft.size():,}")

#train the TFT

# Create a Lightning trainer to handle the training loop.
trainer = pl.Trainer(
    max_epochs=5, #how much the model will go through training prcoess
    accelerator="auto",#what hardware to use
    gradient_clip_val=0.1, #adjustment size
    limit_train_batches=30, #batches per epochs
)

# Train the TFT using the training data
# and evaluate it on the validation data after each epoch.
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)





