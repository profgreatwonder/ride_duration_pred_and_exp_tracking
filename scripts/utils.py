import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error as rmse
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import mlflow
import pickle


def read_dataframe(filename):
    # read the parquet file
    df = pd.read_parquet(filename)
    
    # duration column creation and conversion to minutes
    df_trip_duration = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["trip_duration"] = df_trip_duration
    df.trip_duration = df.trip_duration.apply(lambda td: td.total_seconds() / 60)

    # duration between 1 and 60 minutes
    df = df[(df.trip_duration >= 1) & (df.trip_duration <= 60)]

    # categorical and numerical columns selection
    categorical_cols = ["pulocationid", "dolocationid"]
    # numerical_cols = ["trip_distance"]

    # turning categorical columns to string
    df[categorical_cols] = df[categorical_cols].astype(str)

    return df
