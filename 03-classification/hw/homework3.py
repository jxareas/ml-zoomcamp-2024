# %% Importing libraries

import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# > Note: sometimes your answer doesn't match one of the options exactly.
# > That's fine. Select the option that's closest to your solution.

# %% ### Dataset

# In this homework, we will use the Bank Marketing dataset. Download it from:
# https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
df = pd.read_csv('.//data/bank-full.csv', sep=';')

# Or use the following bash command to download the dataset:
# ```bash
# wget https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
# ```

# We need to take `bank/bank-full.csv` file from the downloaded zip-file.
# In this dataset, our desired target for the classification task will be `y` variable - has the client subscribed a term deposit or not.
print(df['y'].describe())

# ### Features

# For the rest of the homework, you'll need to use only these columns:
# * `age`,
# * `job`,
# * `marital`,
# * `education`,
# * `balance`,
# * `housing`,
# * `contact`,
# * `day`,
# * `month`,
# * `duration`,
# * `campaign`,
# * `pdays`,
# * `previous`,
# * `poutcome`,
# * `y`
target = 'y'
features = [
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
    target
]
df = df[features]

# Check the list
print(features)

# ### Data preparation

# * Select only the features from above.
# * Check if the missing values are presented in the features.
df[features].isnull().sum()
print(f"Is there at least one null in any column => {df[features].isnull().any().any()}")  # No null values

# %% ### Question 1

# What is the most frequent observation (mode) for the column `education`?

# - `unknown`
# - `primary`
# - `secondary`
# - `tertiary`

print(df['education'].mode()[0])  # secondary
df['education'].value_counts()  # secondary is the most frequent observation

# %% ### Question 2

# Create the correlation matrix for the numerical features of your dataset.
numerical = df[features].dtypes[df[features].dtypes != 'object']
numerical = numerical.index.to_list()
# In a correlation matrix, you compute the correlation coefficient between every pair of features.
corr_matrix = df[numerical].corr()
print(corr_matrix)

# Find the two features with the largest absolute correlation (ignoring the diagonal)
corr_matrix_abs = corr_matrix.abs()

top_corr = corr_matrix_abs.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) \
    .stack() \
    .sort_values(ascending=False)

# What are the two features that have the biggest correlation?

# - `age` and `balance`
# - `day` and `campaign`
# - `day` and `pdays`
# - `pdays` and `previous`
print(top_corr[:1])  # `pdays` and `previous` have the highest correlation with ~ 0.45

# %% ### Target encoding

# * Now we want to encode the `y` variable.
# * Let's replace the values `yes`/`no` with `1`/`0`.
df[target] = (df[target] == 'yes').astype(int)

# ### Split the data

# * Split your data in train/val/test sets with 60%/20%/20% distribution.

# * Use Scikit-Learn for that (the `train_test_split` function) and set the seed to `42`.
RANDOM_SEED = 42


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_train, df_val = train_test_split(df_full_train, test_size=0.25,
                                    random_state=RANDOM_SEED)
# * Make sure that the target value `y` is not in your dataframe.
len(df_train), len(df_val), len(df_test)

# Resetting indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Setting targets
y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

del df_train[target], df_val[target], df_test[target]


# %% ### Question 3

# * Calculate the mutual information score between `y` and other categorical variables in the dataset. Use the training set only.
# * Round the scores to 2 decimals using `round(score, 2)`.

def mutual_info_target_score(series):
    return mutual_info_score(series, y_train)


categorical = df.select_dtypes(include=['object']).columns.to_list()

# Which of these variables has the biggest mutual information score?

# - `contact`
# - `education`
# - `housing`
# - `poutcome`
df_train[categorical] \
    .apply(mutual_info_target_score) \
    .sort_values(ascending=False)  # `poutcome` has the highest mutual information score with ~ 0.02 shannon bits

# %% ### Question 4

# * Now let's train a logistic regression.
# * Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
# * Fit the model on the training dataset.
# * To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
# ```python
# model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
# ```
dicts_train = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dicts_train)
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1_000, random_state=RANDOM_SEED)
model.fit(X_train, y_train)
# * Calculate the accuracy on the validation dataset and round it to 2 decimal digits.

dicts_val = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.fit_transform(dicts_val)
y_pred = model.predict(X_val)
accuracy = (y_val == y_pred).mean()
rounded_accuracy = round(accuracy, 2)
# What accuracy did you get?

# - 0.6
# - 0.7
# - 0.8
# - 0.9
pprint(f"{rounded_accuracy=}")  # 0.9 accuracy

# %% ### Question 5

# Let's find the least useful feature using the *feature elimination* technique.

all_features = categorical + numerical
feat_elimination_df = pd.DataFrame(columns=['feature', 'accuracy', 'difference'])

# Setting the accuracy from the prior model (0.9)
original_accuracy = accuracy


# Helper function to append a dictionary to a dataframe
def append_dict_to_df(dataframe, dict_to_append):
    """
        Append a dictionary as a new row to a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to append to.
        dict_to_append (dict): The dictionary containing the new row data.

        Returns:
        pd.DataFrame: The updated DataFrame.

        Example:
        >>> df = pd.DataFrame(columns=['A', 'B'])
        >>> append_dict_to_df(dataframe, {'A': 1, 'B': 2})
        """
    dataframe = pd.concat([dataframe if not dataframe.empty else None, pd.DataFrame.from_records([dict_to_append])])
    return dataframe


# Train a model with all these features (using the same parameters as in Q4).
# Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
for feature in all_features:
    feature_set = [x for x in all_features if x != feature]
    dicts = df_train[feature_set].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)

    # Fit the model on the training dataset.
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
    dicts = df_val[feature_set].to_dict(orient='records')
    X_val = dv.fit_transform(dicts)
    y_pred = model.predict(X_val)

    current_accuracy = (y_val == y_pred).mean()
    # * For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
    difference =  original_accuracy - current_accuracy

    feat_elimination_df = append_dict_to_df(feat_elimination_df, {
        'feature': feature,
        'accuracy': current_accuracy,
        'difference': difference
    })

    print(f"WITHOUT {feature} -> {difference=}")

# > Note: The difference doesn't have to be positive.

# Which of following feature has the smallest difference?
# - `age`
# - `balance`
# - `marital`
# - `previous`
small_features = ['age', 'balance', 'marital', 'previous']
small_features_filter = feat_elimination_df['feature'].isin(small_features)

filtered_feat_elimination_df = feat_elimination_df[small_features_filter]
# Sorting by the smallest difference to the biggest difference from the original accuracy
filtered_feat_elimination_df.sort_values(by='difference', ascending=True) #marital

# %% ### Question 6

# * Now let's train a regularized logistic regression.
# * Let's try the following values of the parameter `C`: `[0.01, 0.1, 1, 10, 100]`.
C_values = [0.01, 0.1, 1, 10, 100]
regularized_results = pd.DataFrame(columns=['C', 'accuracy'])
# * Train models using all the features as in Q4.
all_features = categorical + numerical
# Dictionary Vectorizer
dv = DictVectorizer(sparse=False)
# Converting the df to dictionary format
dicts_train = df_train[all_features].to_dict(orient='records')
dicts_val = df_val[all_features].to_dict(orient='records')
# Fitting the DictVectorized and transforming the dicts to Numpy Arrays
X_train = dv.fit_transform(dicts_train)
X_val = dv.transform(dicts_val)

# * Calculate the accuracy on the validation dataset and round it to 3 decimal digits.
for c in C_values:
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1_000, random_state=42)
    model.fit(X_train, y_train)
    # * Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
    y_pred = model.predict(X_val)
    accuracy = round((y_val == y_pred).mean(), 3)
    print(f"{c}:", accuracy)

    regularized_results = append_dict_to_df(regularized_results, {
        'C': c,
        'accuracy': accuracy
    })

# Which of these `C` leads to the best accuracy on the validation set?

# - 0.01
# - 0.1
# - 1
# - 10
# - 100
regularized_results.sort_values(by='accuracy', ascending=True)  # 0.1

# > **Note**: If there are multiple options, select the smallest `C`.

# ## Submit the results

# * Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2024/homework/hw03
# * If your answer doesn't match options exactly, select the closest one.
