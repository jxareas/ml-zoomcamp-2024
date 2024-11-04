# %% Importing libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV


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

# %% Model Selection

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.status == 'default').astype('int').values
y_val = (df_val.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

del df_train['status'], df_val['status'], df_test['status']

print(df_train)

# %% Dictionary Vectorizer for train dataset

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

# %% XGBoost

features = dv.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=features)

# %% Training the model

xgb_params = {
    # Learning configuration
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    # Train settings
    'objective': 'binary:logistic',
    'nthread': 12,
    # Reproducibility and logging
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain=dtrain, num_boost_round=10)

y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# %% Training the model and adding a watchlist

xgb_params = {
    # Learning configuration
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    # Train settings
    'objective': 'binary:logistic',
    'nthread': 12,
    'eval_metric': 'auc',
    # Reproducibility and logging
    'seed': 1,
    'verbosity': 1,
}
watchlist = [(dtrain, 'train'), (dval, 'validation')]

model = xgb.train(xgb_params, dtrain=dtrain,
                  verbose_eval=5,
                  evals=watchlist, num_boost_round=200)

#%% XGBoost Hyperparameter Tuning via Grid-Search Cross Validation

xgb_model = xgb.XGBClassifier(objective='binary:logistic', nthread=4, seed=1, verbosity=1)

hyperparameter_grid = {
    'eta': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'min_child_weight': [1, 10, 30]
}

k_folds = 5
grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=hyperparameter_grid,
                           scoring='roc_auc',
                           n_jobs=-1,
                           cv=5,
                           verbose=1)

# Fit the Grid Search to the train data
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)
print("Best ROC AUC score: ", grid_search.best_score_)
