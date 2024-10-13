# %% Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Loading data

data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_url)

# %% Data preparation

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = df.dtypes[df.dtypes == 'object'].index.to_list()
for cat in categorical_columns:
    df[cat] = df[cat].str.lower().str.replace(' ', '_')

# Taking a look at the dataframe data types
print(df.dtypes)  # Why is totalcharges of object type?

# Looks like a numerical variable...yet why is it of object type?
print(df['totalcharges'])

# pd.to_numeric(df['totalcharges']) -> Throws an error

df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

# Printing all the total charges now coerced to NaN
print(df[df['totalcharges'].isnull()][['customerid', 'totalcharges']])

df['totalcharges'] = df['totalcharges'].fillna(0)

# Checking is there is any value in total charges that is null
df['totalcharges'].isnull().any()

# Checking the churn variable
print(df['churn'])

df['churn'] = (df['churn'] == 'yes').astype(int)


