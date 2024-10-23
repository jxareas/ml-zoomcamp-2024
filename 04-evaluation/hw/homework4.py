# %% Homework
from pprint import pprint
from random import shuffle

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
from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

# %% Dataset preparation
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
target = 'y'
features = [x for x in df_columns if x != target]

df = pd.read_csv(filepath_or_buffer='./data/bank-full.csv', usecols=df_columns, sep=";")
df['y'] = df['y'].map({'yes': 1, 'no': 0})
# %% Train-test-validation split

# Split the data into 3 parts: train/validation/test with 60%/20%/20% distribution.
# Use `train_test_split` function for that with `random_state=1`.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train['y']
y_test = df_test['y']
y_val = df_val['y']

del df_train['y'], df_test['y'], df_val['y']

# %% Question 1: ROC AUC feature importance

# ROC AUC could also be used to evaluate feature importance of numerical variables.

# Let's do that:

# For each numerical variable, use it as a score (aka prediction)
# and compute the AUC with the `y` variable as the ground truth.
# Use the training dataset for that.
numerical = df_train.select_dtypes(include=['number']).columns.to_list()
auc_scores = {}

# If your AUC is < 0.5, invert this variable by putting "-" in front
# (e.g. `-df_train['engine_hp']`).

# AUC can go below 0.5 if the variable is negatively correlated with the target variable.
# You can change the direction of the correlation by negating this variable
# - then negative correlation becomes positive.
for col in numerical:
    variable = df_train[col]
    auc = roc_auc_score(y_train, variable)
    if auc < 0.5:
        auc = roc_auc_score(y_train, -variable)  # negate if AUC < 0.5

    auc_scores[col] = auc
    print(f'AUC for {col}: {np.round(auc, 3)}')

# Sorting the AUC scores
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
dv = DictVectorizer(sparse=False)
train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
# LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1_000)
model.fit(X_train, y_train)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_pred = model.predict_proba(X_val)[:, 1]
roc_auc_lr = roc_auc_score(y_val, y_pred).round(3)  # 0.89

# What's the AUC of this model on the validation dataset? (round to 3 digits)
# - 0.69
# - 0.79
# - 0.89
# - 0.99
print(f"{roc_auc_lr=}")

# %% Question 3: Precision and Recall

# Now let's compute precision and recall for our model.
precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
# Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01.
# For each threshold, compute precision and recall.
plt.style.use('ggplot')
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision", color='blue')
plt.plot(thresholds, recall[:-1], label="Recall", color='red')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
# Plot them.

# At which threshold do the precision and recall curves intersect?
# - 0.865
# - 0.265
# - 0.465
# - 0.665
differences = np.abs(precision - recall)
idx = np.argmin(differences)
intersection = thresholds[idx]
print(f"Intersection: {intersection.round(3)}")  # 0.265

# %% Question 4: F1 score

# Precision and recall are conflicting - when one grows, the other goes down.
# That's why they are often combined into the F1 score - a metric that takes into account both.

# This is the formula for computing F1:

# F1 = 2 * (P * R) / (P + R)

# Where P is precision and R is recall.

# Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01.
thresholds = np.arange(0, 1.01, 0.01).round(2)
f1_scores = {}

for t in thresholds:
    y_pred_custom = (y_pred >= t).astype(int)
    P = precision_score(y_val, y_pred_custom, zero_division=0.0)
    R = recall_score(y_val, y_pred_custom, zero_division=0.0)
    F1 = 2 * (P * R) / (P + R)
    print(f"F1 score for {t} with f1= 2 * ({P} * {R}) / ({P} + {R} ")
    f1_scores[t] = F1

# At which threshold is F1 maximal?
# - 0.02
# - 0.22
# - 0.42
# - 0.62
sorted_f1_scores = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
best_threshold, max_f1 = sorted_f1_scores[0]
print(f"Max F1 score is {max_f1.round(2)} at threshold {best_threshold}")  # 0.22

# %% Question 5: 5-Fold CV

fixed_c = 1.0


def train(dataframe, target_vector, C=fixed_c):
    dicts = dataframe[features].to_dict(orient='records')

    dict_vectorizer = DictVectorizer(sparse=False)
    x_training = dict_vectorizer.fit_transform(dicts)
    logistic_model = LogisticRegression(solver='liblinear', C=c, max_iter=1000)

    logistic_model.fit(x_training, target_vector)

    return dict_vectorizer, logistic_model


def predict(dataframe, dict_vectorizer, model):
    dicts = dataframe[features].to_dict(orient='records')

    input_vector = dict_vectorizer.transform(dicts)
    predictions = model.predict_proba(input_vector)[:, 1]

    return predictions


# Use the `KFold` class from Scikit-Learn to evaluate our model on 5 different folds:
n_splits = 5
# KFold(n_splits=5, shuffle=True, random_state=1)
kfolds = KFold(n_splits=n_splits, shuffle=True, random_state=1)
auc_scores = []
# Iterate over different folds of `df_full_train`.
for train_idx, val_idx in kfolds.split(df_full_train):
    # Split the data into train and validation.
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train[target].values
    y_val = df_val[target].values

    # Train the model on train with these parameters:
    # LogisticRegression(solver='liblinear', C=1.0, max_iter=1000).
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    # Use AUC to evaluate the model on validation.
    auc_score = roc_auc_score(y_val, y_pred)
    rounded_auc = auc_score.round(3)
    auc_scores.append(auc_score)

# How large is the standard deviation of the scores across different folds?
# - 0.0001
# - 0.006
# - 0.06
# - 0.26
print('\n C=%s : %.3f +- %.3f' % (fixed_c, np.mean(auc_scores), np.std(auc_scores)))
print(f"Standard deviation for scores: {np.std(auc_scores)}")  # 0.006

# %% Question 6: Hyperparameter Tuning

# Now let's use 5-Fold cross-validation to find the best parameter `C`.

# Iterate over the following `C` values: `[0.000001, 0.001, 1]`.
regularization_params = [0.000001, 0.001, 1]
for c in tqdm(regularization_params):
    # Initialize `KFold` with the same parameters as previously.
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    auc_scores = []
    # Use these parameters for the model:
    # LogisticRegression(solver='liblinear', C=C, max_iter=1000).

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train[target].values
        y_val = df_val[target].values

        dv, model = train(df_train, y_train, C=c)
        y_pred = predict(df_val, dv, model)

        auc_score = roc_auc_score(y_val, y_pred)
        rounded_auc = auc_score.round(3)
        auc_scores.append(auc_score)
    # Compute the mean score as well as the std (round the mean and std to 3 decimal digits).
    # Which `C` leads to the best mean score?
    # - 0.000001
    # - 0.001
    # - 1
    print('\n C=%s : %.3f +- %.3f' % (c, np.mean(auc_scores), np.std(auc_scores)))  # C=1

# If you have ties, select the score with the lowest std.
# If you still have ties, select the smallest `C`.

# Submit the results

# Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2024/homework/hw04.
# If your answer doesn't match the options exactly, select the closest one.
