# Databricks notebook source
from prophet import Prophet
import pandas as pd

# COMMAND ----------

train_df = pd.read_csv("/dbfs/FileStore/rafael.pierre/covid/train.csv")
train_df

# COMMAND ----------

def fit(
  train_df: pd.DataFrame,
  country_region: str = "China",
  weekly_seasonality: bool = True,
  daily_seasonality: bool = True
):

  m = Prophet(weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality)
  train_df = (
    train_df
      .query(f"Country_Region == '{country_region}'")
      .loc[:, ["Date", "ConfirmedCases"]]
      .rename(columns = {"Date": "ds", "ConfirmedCases": "y"})
  )

  train_df["ds"] = pd.to_datetime(train_df["ds"])

  # Trauin
  m.fit(train_df)
  return m

def predict(model: Prophet, periods = 30):

  # Predict
  future = model.make_future_dataframe(periods=periods)
  result = model.predict(future)
  return result

# COMMAND ----------

m = fit(train_df)
forecast = predict(m)
m.plot_components(forecast)

# COMMAND ----------


