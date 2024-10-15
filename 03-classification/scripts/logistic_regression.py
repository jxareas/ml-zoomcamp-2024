# %% Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from pprint import pprint

# %% Loading data

data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_url)

# %% Data preparation

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = df.dtypes[df.dtypes == 'object'].index.to_list()
for cat in categorical_columns:
    df[cat] = df[cat].str.lower().str.replace(' ', '_')

# Taking a look at the dataframe data types
print(df.dtypes)  # Why is totalcharges of object type?

# Looks like a numerical variable...yet why is it of object type?
print(df['totalcharges'])

# pd.to_numeric(df['totalcharges']) -> Throws an error

df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

# Printing all the total charges now coerced to NaN
print(df[df['totalcharges'].isnull()][['customerid', 'totalcharges']])

df['totalcharges'] = df['totalcharges'].fillna(0)

# Checking is there is any value in total charges that is null
df['totalcharges'].isnull().any()

# Checking the churn variable
print(df['churn'])

df['churn'] = (df['churn'] == 'yes').astype(int)

# %% Validation framework

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25,
                                    random_state=1)  # 0.25 as it is the 20% of the 80% of the data -> 20/80 ~ 1/4
len(df_train), len(df_val), len(df_test)

# Resetting indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['churn'].values
y_val = df_val['churn'].values
y_test = df_test['churn'].values

del df_train['churn'], df_val['churn'], df_test['churn']

# %% EDA

# Checking for null values
df_full_train.isnull().sum()

df_full_train['churn'].value_counts()
df_full_train['churn'].value_counts(normalize=True)

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies', 'contract',
               'paperlessbilling', 'paymentmethod']
# Take a look at the number of unique values for all the categorical variables
df_full_train[categorical].nunique()

# %% Feature importance: churn rate and risk ratio

df_full_train.head()

global_churn = df_full_train['churn'].mean().round(3)

# Churn rate separated by gender
churn_female = df_full_train[df_full_train['gender'] == 'female']['churn'].mean().round(3)
churn_male = df_full_train[df_full_train['gender'] == 'male']['churn'].mean().round(3)

print(f"{global_churn=}\n{churn_female=}\n{churn_male=}")

# Taking a look at the `partner` variable
df_full_train['partner'].value_counts()

churn_yes_partner = df_full_train[df_full_train['partner'] == 'yes']['churn'].mean().round(3)
churn_no_partner = df_full_train[df_full_train['partner'] == 'no']['churn'].mean().round(3)
print(f"{global_churn=}\n{churn_yes_partner=}\n{churn_no_partner=}")

# %% Risk Ratio

print(f"{churn_no_partner/global_churn=}")
print(f"{churn_yes_partner / global_churn}")

# SELECT
#     gender,
#     AVG(churn),
#     AVG(churn) - global_churn AS diff,
#     AVG(churn) / global_churn AS risk
# FROM
#     data
# GROUP BY
#     gender;
for c in categorical:
    print("-------------------------------------------------------------------")
    print(c.upper())
    df_group = df_full_train.groupby(c)['churn'].agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    print(f"{df_group}\n\n")

# %% Feature Importance : Mutual Information

# Calculating the mutual information for churn and contract, gender and partner variables
mutual_info_score(df_full_train['churn'], df_full_train['contract'])
mutual_info_score(df_full_train['churn'], df_full_train['gender'])
mutual_info_score(df_full_train['churn'], df_full_train['partner'])


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train['churn'])


df_full_train[categorical].apply(mutual_info_churn_score).sort_values(ascending=False)

# %% Feature Importance : Correlation

df_full_train[numerical].corrwith(df_full_train['churn'])

# %% One-hot encoding

# Taking a look at the first 100 elements
print(df_train[['gender', 'contract', 'tenure']].iloc[:100])

# Creating a dictionary out of the former df_train data set
dicts = df_train.iloc[:100].to_dict(orient='records')
pprint(dicts)

# Creating an instance of the dictionary vectorizer
dv = DictVectorizer(sparse=False)

# Make the `DictVectorizer` learn our list of feature names -> indices mappings.
dv.fit(dicts)

# Featureslearned by the dictionary vectorizer
pprint(dv.get_feature_names_out())

# Transform feature->value dicts to array or sparse matrix.
dv.transform(dicts)

# ------------------------------------------------------------------
# Training dataset one-hot encoding
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
pprint(train_dicts[0])

# Creating an instance of the dictionary vectorizer
dv = DictVectorizer(sparse=False)

# Fit and transform for train dataset
X_train = dv.fit_transform(train_dicts)

# Validation dataset one-hot encoding
val_dicts = df_val[categorical + numerical].to_dict(orient='records')
pprint(val_dicts[0])

# Fit and transform for validation dataset
X_val = dv.transform(val_dicts)


# %% Logistic Regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-5, 5, 51)
print(f"{sigmoid(z)=}")

plt.plot(z, sigmoid(z), label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', color='red', linewidth=2)

# Plotting the sigmoid function
plt.title('Sigmoid Function', fontsize=18, weight='bold')
plt.xlabel('Input (z)', fontsize=14)
plt.ylabel(r'Output ($\sigma(z)$)', fontsize=14)
plt.axhline(0, color='gray', linewidth=0.8)  # Horizontal axis line
plt.axvline(0, color='gray', linewidth=0.8)  # Vertical axis line
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid with dashed lines
plt.legend(fontsize=12)
plt.tight_layout()  # Ensures everything fits well within the figure

# Display the plot
plt.show()

# %% Training Logistic Regression

model = LogisticRegression()
# Training the logistic regression model
model.fit(X_train, y_train)

print(model.intercept_[0].round(3))

print(model.coef_.round(3))

model.predict(X_train)

# Outputs a matrix whose column represent P(y=0|X_train) and P(y=1|X_train)
model.predict_proba(X_train).round(3)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = y_pred >= 0.5

# Calculating accuracy
accuracy = (y_val == churn_decision).mean()  # 80% of accuracy
print(f"Accuracy: {(accuracy * 100).round(2)}%")

# %% Model Interpretation


# %% Using the model
