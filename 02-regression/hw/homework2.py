# %% Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
# In this homework, we will use the Laptops price dataset from Kaggle:
# https://www.kaggle.com/datasets/juanmerinobermejo/laptops-price-dataset

# Here's a wget-able link to the dataset:
# !wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv

# The goal of this homework is to create a regression model for predicting the prices (column 'Final Price').

# %% Preparing the dataset
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv')

# First, we'll normalize the names of the columns:
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Now, instead of 'Final Price', we have 'final_price'.
print(df['final_price'])

# Next, use only the following columns:
# - 'ram',
# - 'storage',
# - 'screen',
# - 'final_price'
selected_features = ['ram', 'storage', 'screen', 'final_price']
df_filtered = df[selected_features]
print(df_filtered.head(n=3))

# EDA
# Look at the `final_price` variable. Does it have a long tail?
sns.histplot(data=df, x='final_price', stat='density', fill=True, alpha=0.3)
sns.kdeplot(data=df, x='final_price', color='red')
plt.title(label='Laptop price distribution')
plt.xlabel(xlabel='Price')
plt.show()
# Yes! It does have a long tail as it is skewed to the right
# EDA Interpretation: Most laptops are priced in the lower to mid-range, while a smaller number of more expensive,
# high-end laptops skew the price distribution to the right.

# %% Question 1
# There's one column with missing values. What is it?
# - 'ram'
# - 'storage'
# - 'screen'
# - 'final_price'
nullable_columns = df_filtered.isnull().sum() >= 1
df_filtered.columns[nullable_columns].to_list()  # screen

# %% Question 2
# What's the median (50% percentile) for variable 'ram'?
# - 8
# - 16
# - 24
# - 32
df_filtered['ram'].median()  # 16.0

# %% Prepare and split the dataset
# Shuffle the dataset (the filtered one you created above), use seed 42.
# Split your data into train/val/test sets, with 60%/20%/20% distribution.

# Total number of records
n = len(df_filtered)

# Setting the seed and creating the indices
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
# Use the same code as in the lectures.
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - (n_val + n_test)

# Setting the values via indices and resetting the index
df_train = df_filtered.iloc[idx[:n_train]].reset_index(drop=True)
df_val = df_filtered.iloc[idx[n_train:n_train + n_test]].reset_index(drop=True)
df_test = df_filtered.iloc[idx[n_train + n_test:]].reset_index(drop=True)

# Setting the target variable vector
y_train = df_train['final_price'].values
y_val = df_val['final_price'].values
y_test = df_test['final_price'].values

# Deleting the target variable from the train, validation and testing dataframes
del df_train['final_price'], df_val['final_price'], df_test['final_price']


# %% Linear Regression functions

def prepare_X(df_lr, strategy):
    if strategy not in ['zero', 'mean']:
        raise ValueError("Strategy must be either 'zero' or 'mean'")

    df_lr = df_lr.copy()

    if strategy == 'zero':
        # Fill NaN values with 0
        df_lr = df_lr.fillna(0)
    elif strategy == 'mean':
        # Fill NaN values with the mean of each numeric column
        df_lr['screen'] = df_lr['screen'].fillna(df_lr['screen'].mean())

    X = df_lr.values
    return X


def train_linear_regression(X, y):
    ones_array = np.ones(len(X))
    X = np.column_stack([ones_array, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)

    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]


# For computing the mean, use the training data only!
def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)


# %% Question 3
# We need to deal with missing values for the column from Q1.
# We have two options: fill it with 0 or with the mean of this variable.
X_train_fill_zero = prepare_X(df_lr=df_train, strategy='zero')
X_train_fill_mean = prepare_X(df_lr=df_train, strategy='mean')
# Try both options. For each, train a linear regression model without regularization using the code from the lessons.
w0_zero, w_full_zero = train_linear_regression(X=X_train_fill_zero, y=y_train)
w0_mean, w_full_mean = train_linear_regression(X=X_train_fill_mean, y=y_train)
print(f"Filling with zero: w0={np.round(w0_zero, 2)}, w={np.round(w_full_zero, 2)}")
print(f"Filling with mean: w0={np.round(w0_mean, 2)}, w={np.round(w_full_mean, 2)}")
# Use the validation dataset to evaluate the models and compare the RMSE of each option.
X_val_zero = prepare_X(df_lr=df_val, strategy='zero')
X_val_mean = prepare_X(df_lr=df_val, strategy='mean')
score_fill_zero = rmse(y=y_val, y_pred=w0_zero + X_val_zero.dot(w_full_zero))
score_fill_mean = rmse(y=y_val, y_pred=w0_mean + X_val_mean.dot(w_full_mean))
# Round the RMSE scores to 2 decimal digits using round(score, 2).
score_fill_zero = np.round(score_fill_zero, 2)
score_fill_mean = np.round(score_fill_mean, 2)
# Which option gives better RMSE?
# - With 0
# - With mean
# - Both are equally good
print(f"{score_fill_zero=}")
print(f"{score_fill_mean=}")


# %% Question 4
# Now let's train a regularized linear regression.
# For this question, fill the NAs with 0.
# Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
# Use RMSE to evaluate the model on the validation dataset.
# Round the RMSE scores to 2 decimal digits.
# Which r gives the best RMSE?
# - 0
# - 0.01
# - 1
# - 10
# - 100

# %% Question 5
# We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
# Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
# For each seed, do the train/validation/test split with 60%/20%/20% distribution.
# Fill the missing values with 0 and train a model without regularization.
# For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
# What's the standard deviation of all the scores?
# To compute the standard deviation, use np.std.
# Round the result to 3 decimal digits (round(std, 3)).

# What's the value of std?
# - 19.176
# - 29.176
# - 39.176
# - 49.176

# %% Question 6
# Split the dataset like previously, use seed 9.
# Combine train and validation datasets.
# Fill the missing values with 0 and train a model with r=0.001.
# What's the RMSE on the test dataset?

# Options:
# - 598.60
# - 608.60
# - 618.60
# - 628.60

# Submit the results
# Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2024/homework/hw02
# If your answer doesn't match options exactly, select the closest one.
