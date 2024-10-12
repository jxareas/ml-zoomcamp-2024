#!/usr/bin/env python
# coding: utf-8

# # Preparing the data

# In[137]:


import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv')
df.head(n=5)


# First normalize the names of the columns:

# In[138]:


df.columns = df.columns.str.lower().str.replace(' ', '_')
df.head(n=5)


# Use only the following columns:
# - ram
# - storage
# - screen
# - final_price

# In[139]:


columns_filter = ['ram', 'storage', 'screen', 'final_price']
df_filtered = df[columns_filter]


# # Exploratory Data Analysis
#
# Look ath the final_price variable. Does it have a long tail? It looks like it does.

# In[140]:


sns.histplot(df_filtered.final_price, bins=50)


# # Question 1
#
# There's one column with missing values. What is it?

# In[141]:


df_filtered.isnull().sum()


# **R/** screen

# # Question 2
#
# What's the medina (50% percentile) of variable 'ram'?

# In[142]:


df_filtered.ram.median()


# **R/** 16

# # Prepare and split the dataset

# In[143]:


# Shuffle the dataset (the filtered one you created above), use seed 42
n = len(df_filtered)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

# Split your data in train/val/test, with 60%|20%|20% distribution
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

df_train = df_filtered.iloc[idx[:n_train]].reset_index(drop=True)
df_val = df_filtered.iloc[idx[n_train:n_train + n_test]].reset_index(drop=True)
df_test = df_filtered.iloc[idx[n_train + n_test:]].reset_index(drop=True)

y_train = df_train.final_price.values
y_val = df_val.final_price.values
y_test = df_test.final_price.values

del df_train['final_price']
del df_val['final_price']
del df_test['final_price']


# # Question 3
#
# - We need to deal with missing values for the column from Q1.
# - We have two options: fill it with 0 or with the mean of this variable.
# - Try both options. For each, train a linear regression model without regularization using the code from the lessons.
# - For computing the mean, use the training only!
# - Use the validation dataset to evaluate the models and compare the RMSE of each option.
# - Round the RMSE scores to 2 decimal digits using round(score, 2)
# - Which option gives better RMSE?

# In[144]:


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTXI = np.linalg.inv(XTX)
    w = XTXI.dot(X.T).dot(y)
    return w[0], w[1:]

def rmse(y, y_pred):
    squared_error = (y - y_pred)**2
    mse = np.mean(squared_error)
    return np.sqrt(mse)


# Filling with 0

# In[145]:


X_train = df_train.fillna(0).values

# Train
w0, w = train_linear_regression(X_train, y_train)

# Validate
X_val = df_val.fillna(0).values
y_pred = w0 + X_val.dot(w)
score = rmse(y_val, y_pred)
round(score, 2)


# Filling with mean

# In[146]:


mean_screen = df_train.screen.mean()

X_train = df_train.fillna(mean_screen).values

# Train
w0, w = train_linear_regression(X_train, y_train)

# Validate
X_val = df_val.fillna(mean_screen).values
y_pred = w0 + X_val.dot(w)
score = rmse(y_val, y_pred)
round(score, 2)


# **R/** Filling with 0

# # Question 4
#
# - Now let's train a regularized linear regression.
# - For this question, fill the NAs with 0.
# - Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
# - Use RMSE to evaluate the model on the validation dataset.
# - Round the RMSE scores to 2 decimal digits.
# - Which r gives the best RMSE?
#
# If there are multiple options, select the smallest r.

# In[147]:


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTXI = np.linalg.inv(XTX)
    w = XTXI.dot(X.T).dot(y)
    return w[0], w[1:]

def prepare_X(df):
    df = df.copy()
    X = df.fillna(0).values
    return X


# In[148]:


for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val, y_pred)

    print("r:", r,"w0:",  w0, "score:", round(score, 2))


# **R/** 10

# # Question 5
#
# - We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
# - Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
# - For each seed, do the train/validation/test split with 60%/20%/20% distribution.
# - Fill the missing values with 0 and train a model without regularization.
# - For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
# - What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
# - Round the result to 3 decimal digits (round(std, 3))
#
# What's the value of std?
#
# > Note: Standard deviation shows how different the values are.
# > If it's low, then all values are approximately the same.
# > If it's high, the values are different.
# > If standard deviation of scores is low, then our model is *stable*.

# In[149]:


scores = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df_filtered.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df_filtered.iloc[idx[n_train:n_train + n_test]].reset_index(drop=True)

    y_train = df_train.final_price.values
    y_val = df_val.final_price.values

    del df_train['final_price']
    del df_val['final_price']

    X_train = prepare_X(df_train)
    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)

    scores.append(rmse(y_val, y_pred))

print("Standard Deviation:", round(np.std(scores), 3))


# **R/** 29.176

# # Question 6
#
# - Split the dataset like previously, use seed 9.
# - Combine train and validation datasets.
# - Fill the missing values with 0 and train a model with r=0.001.
# - What's the RMSE on the test dataset?
#
#

# In[152]:


seed = 9

idx = np.arange(n)
np.random.seed(seed)
np.random.shuffle(idx)

df_train = df_filtered.iloc[idx[:n_train]].reset_index(drop=True)
df_val = df_filtered.iloc[idx[n_train:n_train + n_test]].reset_index(drop=True)
df_test = df_filtered.iloc[idx[n_train + n_test:]].reset_index(drop=True)

y_train = df_train.final_price.values
y_val = df_val.final_price.values
y_test = df_test.final_price.values

del df_train['final_price']
del df_val['final_price']
del df_test['final_price']

df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)

X_full_train = prepare_X(df_full_train)
y_full_train = np.concatenate([y_train, y_val])

w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)

score = rmse(y_test, y_pred)
score


# **R/** 608.60
#
