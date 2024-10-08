# %% Importing libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Data Preparation

# dataset_url = './data/cars.csv'
dataset_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
df = pd.read_csv(dataset_url)
df.head()

# Normalizing the data frame column names
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(df.columns)

# Fetching all the string columns AKA those with object as pandas dtype
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
print(string_columns)

# Normalizing the values for all the string columns
for column in string_columns:
    df[column] = df[column].str.lower().str.replace(' ', '_')

df[string_columns].head()

# %% Exploratory Data Analysis

for column in df.columns:
    print(f"Name: {column}")
    print(f"Top five unique values: {df[column].unique()[:5]}")
    print(f"Number of unique: {df[column].nunique()}\n")

sns.histplot(df.msrp, bins=50)
plt.show()

sns.histplot(df.msrp[df.msrp < 100_000], bins=50)
plt.show()

# Variable transform: x -> log(x + 1)
price_logs = np.log1p(df.msrp)

# Plotting the log transform
sns.histplot(price_logs, bins=50)
plt.show()

# Missing values
df.isnull().sum().sort_values(ascending=False)
