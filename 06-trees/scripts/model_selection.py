# %% Importing libraries
from pprint import pprint

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

# %%  Taking a look at the summary statistics
df.describe()

coding = 99999999
for encoded_col in ['income', 'assets', 'debt']:
    df[encoded_col] = df[encoded_col].replace(to_replace=coding, value=np.nan)
    max_value = df[encoded_col].max()

    print(f"NEW MAX for {encoded_col} => {max_value}")

df = df[df['status'] != 'unk'].reset_index(drop=True)

# %% Train-test-validation split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.status == 'default').astype('int').values
y_val = (df_val.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

del df_train['status'], df_val['status'], df_test['status']

# %% Processing the X train, validation and test datasets

# Converting the train dataframe to a list of dictionaries
train_dicts = df_train.to_dict(orient='records')
# Creating the dictionary vectorizer
dv = DictVectorizer(sparse=False)
# Fitting the dictionary vectorizer and transforming the train data
X_train = dv.fit_transform(train_dicts)

# Converting the validation dataframe to a list of dictionaries
val_dicts = df_val.to_dict(orient='records')
# Transforming the validation data
X_val = dv.transform(val_dicts)

# Converting the test dataframe to a list of dictionaries
test_dicts = df_test.to_dict(orient='records')
# Transforming the test data
X_test = dv.transform(test_dicts)

# %% Decision Tree Classifier

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=1)
dt.fit(X_train, y_train)

# %% Random Forest Classifier

rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=1)
rf.fit(X_train, y_train)

# %% XGBoost Classifier
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

features = dv.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=features)

xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=175)

# %% Evaluation using ROC AUC score
evals = []

y_pred_decision_tree = dt.predict_proba(X_val)[:, 1]
roc_auc_decision_tree = roc_auc_score(y_val, y_pred_decision_tree)
evals.append(('Decision Tree', roc_auc_decision_tree))

y_pred_random_forest = rf.predict_proba(X_val)[:, 1]
roc_auc_random_forest = roc_auc_score(y_val, y_pred_random_forest)
evals.append(('Random Forest', roc_auc_random_forest))

y_pred_xgboost = xgb_model.predict(dval)
roc_auc_xgboost = roc_auc_score(y_val, y_pred_xgboost)
evals.append(('XGBoost', roc_auc_xgboost))

evals_df = pd.DataFrame(data=evals, columns=['model', 'roc_auc'])
evals_df = evals_df.sort_values(by='roc_auc', ascending=False).reset_index(drop=True)
print(tabulate(evals_df, headers='keys', tablefmt='grid'))

# %% Setting the full training set

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = (df_full_train['status'] == 'default').astype(int).values
del df_full_train['status']

dicts_full_train = df_full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

# %% Training the final XGBoost model

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names_out().tolist())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out().tolist())

xgb_final_model = xgb.train(xgb_params, dfulltrain, num_boost_round=175)

y_pred_final_xgboost_model = xgb_final_model.predict(dtest)
roc_auc_score(y_test, y_pred_final_xgboost_model)
