import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report

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
    plt.tight_layout()
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

# ***

tmp = df.groupby("Location").apply(lambda x: x.isna().mean())
sns.heatmap(tmp, cmap="Blues")
plt.title("Частка пропусків за локаціями після очищення")
plt.tight_layout()
plt.show()

# ***
# Видаляємо ознаки з великою кількістю пропусків (>35%)
cols_to_drop = ["Sunshine"]
df = df.drop(columns=cols_to_drop)


# Для подальшого аналізу розіб’ємо датасет на окремі вибірки в залежності від типів вхідних даних (числові й категоріальні):
data_num = df.select_dtypes(include=np.number)
data_cat = df.select_dtypes(include=["object", "string"])

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
g.fig.suptitle("Розподіли числових ознак", y=1.02)
g.tight_layout()
plt.show()

print("\nЧислові ознаки:", list(data_num.columns))
print("\nКатегоріальні ознаки:", list(data_cat.columns))


# Конвертуємо колонку 'Date' в формат datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Створюємо нові колонки "Year" та "Month"
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# ВАЖЛИВО: робимо Month категоріальною ознакою
df["Month"] = df["Month"].astype("category")

# Переміщуємо Year у числові ознаки, Month — у категоріальні
# Числові ознаки
data_num = df.select_dtypes(include=[np.number])
# Категоріальні ознаки
data_cat = df.select_dtypes(include=["object", "string"])

# Переміщуємо Month у категоріальні
data_cat["Month"] = df["Month"].astype("category")

# Переміщуємо Year у числові
data_num["Year"] = df["Year"]

print("\nЧислові ознаки:", list(data_num.columns))
print("\nКатегоріальні ознаки:", list(data_cat.columns))

# Знаходимо останній рік спостережень
max_year = df["Year"].max()
print("\nМаксимальний рік у датасеті:", max_year)

#Створюємо булеві маски для train/test

# Об'єкти з останнього року → тест
test_mask = df["Year"] == max_year

# Усі інші роки → тренувальна вибірка
train_mask = df["Year"] < max_year

# Формуємо X_train, X_test, y_train, y_test
#    таргет — RainTomorrow
#    фічі — все інше, включно з Year, Month, але без Date

# Таргет
y = df["RainTomorrow"]

# Фічі (без таргета, без Date)
X = df.drop(columns=["RainTomorrow"])

# Train / Test розбиття за роками
X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nФорма X_train:", X_train.shape)
print("Форма y_train:", y_train.shape)
print("Форма X_test:", X_test.shape)
print("Форма y_test:", y_test.shape)


# Імпутація пропущених значень (SimpleImputer)
#    числові ознаки → заповнюємо медіаною
#    категоріальні ознаки → заповнюємо значенням "Unknown"

# Імпутери
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Визначаємо числові та категоріальні колонки
#num_cols = X_train.select_dtypes(include=[np.number]).columns
#cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns

# Імпутуємо TRAIN
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])

# Імпутуємо TEST (тільки transform!)
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])


# Масштабування числових ознак (StandardScaler)
# масштабуємо тільки числові ознаки, і тільки на TRAIN робимо .fit(), а на TEST — .transform().
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# Кодування категоріальних ознак (OneHotEncoder)
#    використовуємо handle_unknown="ignore"
#    кодуємо тільки TRAIN через .fit_transform()
#    TEST — через .transform()
#    після кодування об’єднуємо числові та категоріальні ознаки назад

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Кодуємо TRAIN
encoded_train = encoder.fit_transform(X_train[cat_cols])

# Кодуємо TEST
encoded_test = encoder.transform(X_test[cat_cols])

# Перетворюємо у DataFrame
encoded_train_df = pd.DataFrame(encoded_train, index=X_train.index, columns=encoder.get_feature_names_out(cat_cols))
encoded_test_df = pd.DataFrame(encoded_test, index=X_test.index, columns=encoder.get_feature_names_out(cat_cols))

# Об’єднуємо числові + категоріальні
X_train_final = pd.concat([X_train[num_cols], encoded_train_df], axis=1)
X_test_final = pd.concat([X_test[num_cols], encoded_test_df], axis=1)

print("\nФінальна форма X_train:", X_train_final.shape)
print("Фінальна форма X_test:", X_test_final.shape)


# Експерименти з solver
solvers = ["liblinear", "lbfgs", "saga", "newton-cg", "sag"]

results = {}

for solver in solvers:
    print(f"\n=== Навчання LogisticRegression (solver = {solver}) ===")
    
    model = LogisticRegression(
        solver=solver,
        max_iter=500,
        n_jobs=-1
    )
    
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    
    acc = accuracy_score(y_test, y_pred)
    results[solver] = acc
    
    print(f"Точність (accuracy): {acc:.4f}")
    print("Короткий звіт класифікації:")
    print(classification_report(y_test, y_pred))

# Розрахунок метрик нової моделі
print("\n=== Метрики LogisticRegression ===")
print(classification_report(y_test, y_pred))
