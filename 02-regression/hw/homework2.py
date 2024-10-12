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
print(f"{score_fill_zero=}")  # Filling with zero has a lower RMSE, hence it is better
print(f"{score_fill_mean=}")


# %% Regularized Linear Regression

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTXI = np.linalg.inv(XTX)
    w = XTXI.dot(X.T).dot(y)
    return w[0], w[1:]


# %% Question 4
# Now let's train a regularized linear regression.
# For this question, fill the NAs with 0.
X_train = prepare_X(df_lr=df_train, strategy='zero')
X_val = prepare_X(df_lr=df_val, strategy='zero')
# Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
r = [0, 0.01, 0.1, 1, 5, 10, 100]
rmses = {}
for alpha in r:
    w0, w = train_linear_regression_reg(X=X_train, y=y_train, r=alpha)
    rmses[alpha] = rmse(y=y_val, y_pred=w0 + X_val.dot(w))
# Use RMSE to evaluate the model on the validation dataset.
sorted_rmses = {key: np.round(value, 2) for key, value in sorted(rmses.items(), key=lambda item: item[1])}
# Round the RMSE scores to 2 decimal digits.
# Which r gives the best RMSE?
# - 0
# - 0.01
# - 1
# - 10
# - 100
sorted_rmses  # Answer is 10, from the options.

# %% Question 5
# We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
# Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = {}
target_variable = 'final_price'
# For each seed, do the train/validation/test split with 60%/20%/20% distribution.
for seed in seeds:
    idx = np.arange(n)
    np.random.seed(seed)
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
    y_train = df_train[target_variable].values
    y_val = df_val[target_variable].values
    y_test = df_test[target_variable].values

    # Deleting the target variable from the train, validation and testing dataframes
    del df_train[target_variable], df_val[target_variable], df_test[target_variable]

    # Fill the missing values with 0 and train a model without regularization.
    X_train = prepare_X(df_lr=df_train, strategy='zero')
    w0, w = train_linear_regression(X=X_train, y=y_train)

    X_val = prepare_X(df_lr=df_val, strategy='zero')
    y_pred = w0 + X_val.dot(w)
    # For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
    score = rmse(y=y_val, y_pred=y_pred)
    scores[seed] = score

# What's the standard deviation of all the scores?
# To compute the standard deviation, use np.std.
score_values = [x for x in scores.values()]
# Round the result to 3 decimal digits (round(std, 3)).
score_std_dev = np.std(score_values).round(3)

# What's the value of std?
# - 19.176
# - 29.176
# - 39.176
# - 49.176
print(f"{score_std_dev=}")  # 29.176

# %% Question 6
# Split the dataset like previously, use seed 9.
np.random.seed(9)
# Combine train and validation datasets.
df_full = pd.concat([df_train, df_val])
y_full = np.concatenate([y_train, y_val])

# Fill the missing values with 0 and train a model with r=0.001.
X_full = prepare_X(df_lr=df_full, strategy='zero')
w0, w = train_linear_regression_reg(X=X_full, y=y_full, r=0.001)

X_test = prepare_X(df_lr=df_test, strategy='zero')
# What's the RMSE on the test dataset?
# Options:
# - 598.60
# - 608.60
# - 618.60
# - 628.60
test_rmse = rmse(y=y_test, y_pred=w0 + X_test.dot(w))
print(f"{test_rmse=}") # 608.60

# Submit the results
# Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2024/homework/hw02
# If your answer doesn't match options exactly, select the closest one.
