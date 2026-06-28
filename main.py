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
    sep=r"\s+",
    header=None,
    skiprows=27,
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