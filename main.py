import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


# 2. Завантаження
DATA_PATH = Path("data/raw/weatherAUS.csv")

df = pd.read_csv(
    DATA_PATH,
    header=1,
    encoding="latin1"
)

# columns TBD here
df.columns = [
    "Date", "Location", "MinTemp", "MaxTemp",
    "Rainfall", "Evaporation", "Sunshine", 
    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", 
    "RainToday", "RainTomorrow"
]

# 2.1. Перевіримо завантаження
print(df.head())
print("\nФорма датасету:", df.shape)
print(df.dtypes)
print("")

# Перевірка на наявність пропущених значень
empty_vals = df.isna().mean().sort_values(ascending=False)
print("Пропущені значення (від найбільшого до найменшого):")
print(empty_vals)

#df2 = df.drop(["Location", "Date"], axis=1)
#
#tmp = (
#    df2.groupby(df["Location"])
#       .apply(lambda x: x.isna().mean())
#)

tmp = (
    df.drop(["Date"], axis=1)
      .groupby("Location")
      .apply(lambda x: x.isna().mean())
)


plt.figure(figsize=(9, 13))

ax = sns.heatmap(tmp,
    cmap='Blues',
    linewidth=0.5,
    square=True,
    cbar_kws=dict(
        location="bottom",
        pad=0.01,
        shrink=0.25)
    )

ax.xaxis.tick_top()
ax.tick_params(axis='x', labelrotation=90)

plt.show()
