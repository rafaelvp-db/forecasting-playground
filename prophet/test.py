# Databricks notebook source
import prophet
import pandas as pd
 
df = pd.DataFrame({"ds": [1.0, 2.0, 3.0, 4.0], "y": [0.1, 0.2, 0.3, 0.4]})
model = prophet.Prophet()
model.fit(df)

# COMMAND ----------


