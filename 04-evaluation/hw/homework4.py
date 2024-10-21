# %% Homework
from pprint import pprint

# Note: sometimes your answer doesn't match one of
# the options exactly. That's fine.
# Select the option that's closest to your solution.

# In this homework, we will use the Bank Marketing dataset.
# Download it from here: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

# You can do it with `wget`:

# bash command:
# wget https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
# unzip bank+marketing.zip
# unzip bank.zip

# We need `bank-full.csv`.

# In this dataset, the target variable is `y` - has the client subscribed to a term deposit or not.

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# %% # Dataset preparation
# For the rest of the homework, you'll need to use only these columns:

# 'age',
# 'job',
# 'marital',
# 'education',
# 'balance',
# 'housing',
# 'contact',
# 'day',
# 'month',
# 'duration',
# 'campaign',
# 'pdays',
# 'previous',
# 'poutcome',
# 'y'

df_columns = [
    'age',
    'job',
    'marital',
    'education',
    'balance',
    'housing',
    'contact',
    'day',
    'month',
    'duration',
    'campaign',
    'pdays',
    'previous',
    'poutcome',
    'y'
]
df = pd.read_csv(filepath_or_buffer='./data/bank-full.csv', usecols=df_columns, sep=";")

# Split the data into 3 parts: train/validation/test with 60%/20%/20% distribution.
# Use `train_test_split` function for that with `random_state=1`.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# %% Question 1: ROC AUC feature importance

# ROC AUC could also be used to evaluate feature importance of numerical variables.

# Let's do that:

# For each numerical variable, use it as a score (aka prediction)
# and compute the AUC with the `y` variable as the ground truth.
numerical = df_train.select_dtypes(include=['number']).columns.to_list()
y = df_train['y'].map({'yes': 1, 'no': 0})
auc_scores = {}

# Use the training dataset for that.
# If your AUC is < 0.5, invert this variable by putting "-" in front
# (e.g. `-df_train['engine_hp']`).

# AUC can go below 0.5 if the variable is negatively correlated with the target variable.
# You can change the direction of the correlation by negating this variable
# - then negative correlation becomes positive.
for col in numerical:
    variable = df_train[col]
    auc = roc_auc_score(y, variable)
    if auc < 0.5:
        auc = roc_auc_score(y, -variable)  # negate if AUC < 0.5

    auc_scores[col] = auc
    print(f'AUC for {col}: {np.round(auc, 3)}')

# sorting the AUC scors
sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
rounded_auc_scores = [(name, round(value, 3)) for name, value in sorted_auc_scores]

# Which numerical variable (among the following 4) has the highest AUC?

# - balance
# - day
# - duration
# - previous
selected_vars = ['balance', 'day', 'duration', 'previous']
[(name, value) for name, value in rounded_auc_scores if name in selected_vars]  # duration

# %% Question 2: Training the model

# Apply one-hot-encoding using `DictVectorizer` and train the logistic regression with these parameters:

# LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)

# What's the AUC of this model on the validation dataset? (round to 3 digits)

# - 0.69
# - 0.79
# - 0.89
# - 0.99


# %% Question 3: Precision and Recall

# Now let's compute precision and recall for our model.

# Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01.
# For each threshold, compute precision and recall.
# Plot them.

# At which threshold do the precision and recall curves intersect?

# - 0.265
# - 0.465
# - 0.665
# - 0.865


# %% Question 4: F1 score

# Precision and recall are conflicting - when one grows, the other goes down.
# That's why they are often combined into the F1 score - a metric that takes into account both.

# This is the formula for computing F1:

# F1 = 2 * (P * R) / (P + R)

# Where P is precision and R is recall.

# Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01.

# At which threshold is F1 maximal?

# - 0.02
# - 0.22
# - 0.42
# - 0.62


# %% Question 5: 5-Fold CV

# Use the `KFold` class from Scikit-Learn to evaluate our model on 5 different folds:

# KFold(n_splits=5, shuffle=True, random_state=1)

# Iterate over different folds of `df_full_train`.
# Split the data into train and validation.
# Train the model on train with these parameters:
# LogisticRegression(solver='liblinear', C=1.0, max_iter=1000).
# Use AUC to evaluate the model on validation.

# How large is the standard deviation of the scores across different folds?

# - 0.0001
# - 0.006
# - 0.06
# - 0.26


# %% Question 6: Hyperparameter Tuning

# Now let's use 5-Fold cross-validation to find the best parameter `C`.

# Iterate over the following `C` values: `[0.000001, 0.001, 1]`.
# Initialize `KFold` with the same parameters as previously.
# Use these parameters for the model:
# LogisticRegression(solver='liblinear', C=C, max_iter=1000).

# Compute the mean score as well as the std (round the mean and std to 3 decimal digits).

# Which `C` leads to the best mean score?

# - 0.000001
# - 0.001
# - 1

# If you have ties, select the score with the lowest std.
# If you still have ties, select the smallest `C`.

# Submit the results

# Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2024/homework/hw04.
# If your answer doesn't match the options exactly, select the closest one.
