# %% Libraries
from pprint import pprint

# Note: sometimes your answer doesn't match one of the options exactly.
# That's fine. Select the option that's closest to your solution.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
import seaborn as sns

# %% Loading the dataset

# In this homework, we will use the Students Performance in 2024 JAMB dataset
# from Kaggle (https://www.kaggle.com/datasets/idowuadamo/students-performance-in-2024-jamb).

# Here's a wget-able link:
# wget https://github.com/alexeygrigorev/datasets/raw/refs/heads/master/jamb_exam_results.csv
df = pd.read_csv('./data/students-performance-2024.csv')

# The goal of this homework is to create a regression model for predicting the
# performance of students on a standardized test (column 'JAMB_Score').

# %% Preparing the dataset

# First, let's make the names lowercase:
# df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = df.columns.str.lower().str.replace(' ', '_')
# Preparation:

# - Remove the 'student_id' column.
del df['student_id']
# - Fill missing values with zeros.
df = df.fillna(0)

target = 'jamb_score'

# %% Train-test-validation split

# - Do train/validation/test split with 60%/20%/20% distribution.
# - Use the train_test_split function and set the random_state parameter to 1.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train[target].values
y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

del df_train[target], df_test[target], df_val[target], df_full_train[target]

# - Use DictVectorizer(sparse=True) to turn the dataframes into matrices.
dv = DictVectorizer(sparse=True)

train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)

del train_dicts, val_dicts, test_dicts

# %% Question 1

# Let's train a decision tree regressor to predict the 'jamb_score' variable.

# - Train a model with max_depth=1.
dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)
dt_text = export_text(decision_tree=dt, feature_names=dv.get_feature_names_out())

# Which feature is used for splitting the data?

# - study_hours_per_week
# - attendance_rate
# - teacher_quality
# - distance_to_school
print(dt_text)  # study_hours_per_week

# %% Question 2

# Train a random forest regressor with these parameters:

# - n_estimators=10
# - random_state=1
# - n_jobs=-1 (optional - to make training faster)
rf = RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)

# What's the RMSE of this model on the validation data?

# - 22.13
# - 42.13
# - 62.13
# - 82.12
rmse(y_val, y_pred).__round__(3)  # 42.13

# %% Question 3

# Now let's experiment with the n_estimators parameter
scores = []

# - Try different values of this parameter from 10 to 200 with step 10.
for n in range(10, 201, 10):
    # - Set random_state to 1.
    rf = RandomForestRegressor(n_estimators=n, n_jobs=-1, random_state=1)
    # - Evaluate the model on the validation dataset.
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    val_rmse = rmse(y_val, y_pred)
    scores.append((n, val_rmse))

# After which value of n_estimators does RMSE stop improving?
# Consider 3 decimal places for calculating the answer.
df_scores = pd.DataFrame(data=[(x, y.round(3)) for (x, y) in scores],
                         columns=['n_estimators', 'rmse'])

# - 10
# - 25
# - 80
# - 200
plt.style.use('ggplot')
sns.lineplot(data=df_scores, x='n_estimators', y='rmse')
plt.xticks(range(0, df_scores['n_estimators'].max() + 1, 20))

plt.title('When does RMSE stop improving?')
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.show()  # Approx at 80, after that it starts increasing, which indicates a worsening in performance

# %% Question 4

# Let's select the best max_depth:

# - Try different values of max_depth: [10, 15, 20, 25]
depths = [10, 15, 20, 25]
estimator_range = range(10, 201, 10)
mean_scores = {}
# - For each of these values,
for max_depth in depths:
    #   - try different values of n_estimators from 10 till 200 (with step 10)
    scores = []
    for n_estimators in estimator_range:
        # - Fix the random seed: random_state=1
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)
        val_rmse = rmse(y_val, y_pred)
        print(f"{max_depth=} & {n_estimators=} => {val_rmse=}")
        scores.append(val_rmse)
    #   - calculate the mean RMSE
    mean_score = np.mean(scores)
    print(f"{max_depth=} => {mean_score}")
    mean_scores[max_depth] = mean_score

# What's the best max_depth, using the mean RMSE?

# - 10
# - 15
# - 20
# - 25
pprint(mean_scores)  # 10 has the lowest mean RMSE

# %% Question 5

# We can extract feature importance information from tree-based models.

# At each step of the decision tree learning algorithm, it finds the best split.
# When doing it, we can calculate "gain" - the reduction in impurity before and after the split.
# This gain is quite useful in understanding what are the important features for tree-based models.

# In Scikit-Learn, tree-based models contain this information in the
# feature_importances_ field.

# For this homework question, we'll find the most important feature:

# - Train the model with these parameters:
#   - n_estimators=10,
#   - max_depth=20,
#   - random_state=1,
#   - n_jobs=-1 (optional)
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
# - Get the feature importance information from this model
feature_importance = rf.feature_importances_
# Getting the feature names
feature_names = dv.get_feature_names_out()
# All features, with their name and importance
feature_list = [(x, y) for x, y in zip(feature_names, feature_importance)]
df_features = pd.DataFrame(feature_list, columns=['name', 'importance'])
# What's the most important feature (among these 4)?
# - study_hours_per_week
# - attendance_rate
# - distance_to_school
# - teacher_quality
criteria = ['study_hours_per_week', 'attendance_rate', 'distance_to_school', 'teacher_quality']
df_features[df_features['name'].isin(criteria)].sort_values(by='importance', ascending=False)  # study_hours_per_week

# %% Question 6

# Now let's train an XGBoost model! For this question, we'll tune the eta parameter:

# - Install XGBoost
# - Create DMatrix for train and validation
# - Create a watchlist
# - Train a model with these parameters for 100 rounds:
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
watchlist = [(dtrain, 'train'), (dval, 'validation')]

# xgb_params = {
#     'eta': 0.3,
#     'max_depth': 6,
#     'min_child_weight': 1,
#     'objective': 'reg:squarederror',
#     'nthread': 8,
#     'seed': 1,
#     'verbosity': 1,
# }
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist)
y_val_pred_eta_1 = model.predict(dval)
rmse_eta_1 = rmse(y_val, y_val_pred_eta_1)
print(f"RMSE for eta=0.3: {rmse_eta_1:.3f}")

# Now change eta from 0.3 to 0.1.
xgb_params['eta'] = 0.1

# Train the model again with the new eta
model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist)

# Predict on validation set
y_val_pred_eta_2 = model.predict(dval)

# Calculate RMSE for eta=0.1
rmse_eta_2 = rmse(y_val, y_val_pred_eta_2)
print(f"RMSE for eta=0.1: {rmse_eta_2:.3f}")

# Which eta leads to the best RMSE score on the validation dataset?

# - 0.3
# - 0.1
# - Both give equal value
# Compare RMSE results and determine which eta leads to the best score
if rmse_eta_1 < rmse_eta_2:
    best_eta = 0.3
    print(f"{best_eta=} with RMSE: {rmse_eta_1}")
elif rmse_eta_2 < rmse_eta_1:
    best_eta = 0.1
    print(f"{best_eta=} with RMSE: {rmse_eta_2}")
else:
    best_eta = 'Both give equal value'


# Submit the results

# - Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2024/homework/hw06
# - If your answer doesn't match options exactly, select the closest one
