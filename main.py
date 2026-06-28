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


# -----------------------------
# 1. Завантаження та перевірка датасету
# -----------------------------

# 1.1. Завантажимо датасет
DATA_PATH = Path("data/raw/weatherAUS.csv")

df = pd.read_csv(
    DATA_PATH,
    header=1,
    encoding="latin1"
)

df.columns = [
    "Date", "Location", "MinTemp", "MaxTemp",
    "Rainfall", "Evaporation", "Sunshine", 
    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", 
    "RainToday", "RainTomorrow"
]

# 1.2. Перевіримо завантаження
print(df.head())
print("\nФорма датасету:", df.shape)
print(df.dtypes)
print("")

# -----------------------------
# 2. Перевірка на наявність пропущених значень
# -----------------------------

# 2.1. Перевіримо пропущені значення
empty_vals = df.isna().mean().sort_values(ascending=False)
print("\nПропущені значення (від найбільшого до найменшого):")
print(empty_vals)

# 2.2. Візуалізація пропущених значень

def plot_missing_by_location(df, title="Пропуски за локаціями"):
    """
    Будує heatmap пропусків для кожної локації.
    df — датафрейм після завантаження або після очищення.
    """

    tmp = (
        df.drop(["Date"], axis=1, errors="ignore")
          .groupby("Location")
          .apply(lambda x: x.isna().mean())
    )

    plt.figure(figsize=(9, 13))
    ax = sns.heatmap(
        tmp,
        cmap='Blues',
        linewidth=0.5,
        square=True,
        cbar_kws=dict(location="bottom", pad=0.01, shrink=0.25)
    )

    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelrotation=90)
    plt.title(title)
    plt.show()

plot_missing_by_location(df, title="Пропуски ДО очищення")


# -----------------------------
# 3. Обробка пропущених значень
# -----------------------------

# 3.1. Таргет RainTomorrow — видаляємо пропуски
df = df.dropna(subset=["RainTomorrow"])

# 3.2. Категоріальні ознаки → "Unknown"
cat_cols = [
    "WindGustDir", "WindDir9am", "WindDir3pm",
    "RainToday"
]

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# 3.3. Числові ознаки з невеликою кількістю пропусків → медіана
num_small = [
    "MinTemp", "MaxTemp", "Rainfall",
    "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Temp9am", "Temp3pm"
]

for col in num_small:
    df[col] = df[col].fillna(df[col].median())

# 3.4. Числові ознаки з великими пропусками → медіана по Location
heavy_cols = [
    "Evaporation", "Sunshine",
    "Cloud9am", "Cloud3pm"
]

for col in heavy_cols:
    df[col] = df.groupby("Location")[col].transform(
        lambda x: x.fillna(x.median())
    )

# 3.5. Pressure9am / Pressure3pm → медіана по Location + Month- втрачається багато записів
#df["Month"] = pd.to_datetime(df["Date"]).dt.month 
press_cols = ["Pressure9am", "Pressure3pm"]
#
#for col in press_cols:
#    df[col] = df.groupby(["Location", "Month"])[col].transform(
#        lambda x: x.fillna(x.median())
#    )
# Натомість пробуємо альтернативу:
# 3.5. Pressure → медіана по Location (м’яка імпутація)
for col in press_cols:
    df[col] = df.groupby("Location")[col].transform(
        lambda x: x.fillna(x.median())
    )

# 3.6. Якщо після всіх кроків залишилися пропуски — видаляємо
# Видаляємо тільки ті рядки, де пропуски залишилися у критичних колонках (де імпутація не спрацювала)
critical_cols = ["Pressure9am", "Pressure3pm"]
df = df.dropna(subset=critical_cols)

# Також можна видалити спостереження, для яких відсутня цільова мітка (пропуски в колонці RainTomorrow) - але чи має це сенс після імпутації?
#df = df[df.columns[df.isna().mean().lt(0.35)]]
#df = df.dropna(subset="RainTomorrow")

# 3.7. Перевірка (на очищеному df)
print("\nПропуски після очищення:", df.isna().sum().sum())
print("\nФорма датасету після очищення:", df.shape)

plot_missing_by_location(df, title="Пропуски ПІСЛЯ очищення")


# Для подальшого аналізу розіб’ємо датасет на окремі вибірки в залежності від типів вхідних даних (числові й категоріальні):
data_num = df.select_dtypes(include=np.number)
data_cat = df.select_dtypes(include='object')

# Розглянемо розподіли числових ознак
melted = data_num.melt()
 
g = sns.FacetGrid(melted,
        col='variable',
           col_wrap=4,
           sharex=False,
           sharey=False,
           aspect=1.25)
g.map(sns.histplot, 'value') 
g.set_titles(col_template='{col_name}')
g.tight_layout()
plt.show()

