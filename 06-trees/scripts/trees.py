# %% Importing libraries
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# %% Loading data

data_path = './data/credit-scoring-dataset.csv'
df = pd.read_csv(filepath_or_buffer=data_path)
df.head()

# %% Preparing the data

# Lowercasing the column names
df.columns = df.columns.str.lower()

# Decoding categorical variables
df['status'] = df.status.map({
    0: 'unk',
    1: 'ok',
    2: 'default'
})

df['home'] = df.home.map({
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
})

df['marital'] = df.marital.map({
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
})

df['records'] = df.records.map({
    1: 'no',
    2: 'yes',
    0: 'unk'
})

df['job'] = df.job.map({
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
})

# Taking a look at the summary statistics
df.describe()

coding = 99999999
for encoded_col in ['income', 'assets', 'debt']:
    df[encoded_col] = df[encoded_col].replace(to_replace=coding, value=np.nan)
    max_value = df[encoded_col].max()

    print(f"NEW MAX for {encoded_col} => {max_value}")

df = df[df['status'] != 'unk'].reset_index(drop=True)

# %% Model Selection

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = (df_train['status'] == 'default').astype(int).values
y_val = (df_val['status'] == 'default').astype(int).values
y_test = (df_test['status'] == 'default').astype(int).values

del df_train['status'], df_val['status'], df_test['status']

print(df_train)
