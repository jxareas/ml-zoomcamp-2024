# %% Importing libraries
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

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

# %% Train-Test-Validation split

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


# %% Decision Trees

def assess_risk(client):
    """
    Example of a decision tree
    :param client: Dataset observation from the credit-scoring df
    :return: credit scoring status, either 'ok' or 'default'
    """
    if (client['records'] == 'yes'):
        if (client['job'] == 'parttime'):
            return 'default'
        else:
            return 'ok'
    else:
        if (client['assets'] > 6_000):
            return 'ok'
        else:
            return 'default'


xi = df_train.iloc[0].to_dict()

# Assessing the risk for a particular client
assess_risk(xi)

# %% Dictionary Vectorizer for train dataset

# Converting the train dataframe to a list of dictionaries
train_dicts = df_train.to_dict(orient='records')
# Creating the dictionary vectorizer
dv = DictVectorizer(sparse=False)
# Fitting the dictionary vectorizer and transforming the train data
X_train = dv.fit_transform(train_dicts)
# Feature names
dv.get_feature_names_out()

# %% Fitting the DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# %% AUC score

# Train-AUC
y_pred_train = dt.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, y_pred_train)
print(f"{train_auc=}")

# Converting the validation dataframe to a list of dictionaries
val_dict = df_val.fillna(0).to_dict(orient='records')
# Transforming the validation data
X_val = dv.transform(val_dict)

# Val-AUC
y_pred_val = dt.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_pred_val)
print(f"{val_auc=}")

# Decision Tree rules
print(export_text(dt, feature_names=dv.get_feature_names_out()))

# %% Decision Tree Learning Algorithm

data = [
    [8000, 'default'],
    [2000, 'default'],
    [0, 'default'],
    [5000, 'ok'],
    [5000, 'ok'],
    [4000, 'ok'],
    [9000, 'ok'],
    [3000, 'default'],
]

df_example = pd.DataFrame(data, columns=['assets', 'status'])
df_example

thresholds = [2_000, 3_000, 4_000, 5_000, 8_000]
for t in thresholds:
    df_left = df_example[df_example['assets'] <= t]
    df_right = df_example[df_example['assets'] > t]
    print("-----------------------------------")
    print(f"For {t=}:")
    print("DF LEFT")
    print(df_left)
    print(df_left['status'].value_counts(normalize=True))
    print("DF RIGHT")
    print(df_right)
    print(df_right['status'].value_counts(normalize=True))

# %% Decision Trees Hyperparameter Tuning

for max_depth in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred).round(2)

    print(f"{max_depth=} => {auc_score=}")

scores = []
for max_depth in [4, 5, 6, 7, 10, 15, 20, None]:
    for min_samples_leaf in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred)
        scores.append((max_depth, min_samples_leaf, auc_score))
        print(f"Depth={max_depth}, Leaf={min_samples_leaf} => AUC={auc_score.__round__(3)}")

df_scores = pd.DataFrame(scores, columns=['max_depth', 'min_samples_leaf', 'auc_score'])
df_scores.sort_values('auc_score', ascending=False)

# using the pivot function
df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns='max_depth', values='auc_score')

# %% Creating a heatmap

sns.heatmap(
    df_scores_pivot,
    annot=True,
    fmt=".3f",
    cbar_kws={'label': '', 'shrink': 0.8},
    linewidths=0.3,
    linecolor='lightgray',
)

# Labels
plt.title('AUC Scores for Decision Tree')
plt.xlabel('Max Depth')
plt.ylabel('Min Samples Leaf')

# Rendering the plot
plt.show()

# %% Training a model with the chosen max_depth and min_samples_leaf

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

# %% Ensembles and Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)

scores = []
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((n, auc))

df_scores = pd.DataFrame(data=scores, columns=['n_estimators', 'auc'])

# %% Plotting n_estimators vs roc_auc_score
plt.style.use('ggplot')
plt.figure(figsize=(8, 6))

sns.lineplot(x=df_scores['n_estimators'], y=df_scores['auc'])

# Set the title with the monospace font applied to the "code-like" parts
monospace = FontProperties(family='monospace', size=12)
plt.title("n_estimators vs roc_auc_score", fontproperties=monospace, weight='bold')
plt.suptitle("Impact of estimators on Random Forest ROC AUC", color='gray')
plt.xlabel('n_estimators', fontproperties=monospace, color='black')
plt.ylabel('roc_auc_score', fontproperties=monospace, color='black')

plt.savefig('./charts/m6_n_estimators_v_roc_auc.png', format='png')
plt.show()

# %% Tuning max depth

scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((d, n, auc))

df_scores = pd.DataFrame(data=scores, columns=['max_depth', 'n_estimators', 'auc'])
df_scores.head()

# %% Plotting n_estimators vs roc_auc by max_depth

plt.style.use('bmh')
plt.figure(figsize=(8, 6))
for s in [5, 10, 15]:
    df_subset = df_scores[df_scores['max_depth'] == s]
    sns.lineplot(x=df_subset['n_estimators'], y=df_subset['auc'], label=f"max_depth={s}")

plt.title('Estimators vs ROC AUC')
plt.xlabel('Number of estimators')
plt.ylabel('ROC AUC')
plt.legend()
# plt.savefig('./charts/m6_n_estimators_v_roc_auc_by_max_depth.png', format='png')
plt.show()

# %% Tuning min_samples_leaf

scores = []
fixed_max_depth = 10

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=fixed_max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((s, n, auc))

df_scores = pd.DataFrame(data=scores, columns=['min_samples_leaf', 'n_estimators', 'auc'])
df_scores.head()

fixed_min_samples_leaf = 3
fixed_n_estimators = 100
# %% Plotting n_estimators vs roc_auc by min_samples_leaf

plt.style.use('bmh')
plt.figure(figsize=(8, 6))
for s in [1, 3, 5, 10, 50]:
    df_subset = df_scores[df_scores['min_samples_leaf'] == s]
    sns.lineplot(x=df_subset['n_estimators'], y=df_subset['auc'], label=f"min_samples_leaf={s}")

plt.title('Estimators vs ROC AUC')
plt.xlabel('Number of estimators')
plt.ylabel('ROC AUC')
plt.legend()
plt.savefig('./charts/m6_n_estimators_v_roc_auc_by_min_samples_leaf.png', format='png')
plt.show()

# %% Training the final Random Forest Model

rf = RandomForestClassifier(n_estimators=fixed_n_estimators,
                            max_depth=fixed_max_depth,
                            min_samples_leaf=fixed_min_samples_leaf,
                            n_jobs=-1)
rf.fit(X_train, y_train)
