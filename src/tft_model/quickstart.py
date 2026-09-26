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
print(training)
print("Number of training samples:", len(training)) #training windows created

x, y = training[0]
print(x.keys())
print(y)

