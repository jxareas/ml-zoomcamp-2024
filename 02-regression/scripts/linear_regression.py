# %% Importing libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Data Preparation

# dataset_url = './data/cars.csv'
dataset_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
df = pd.read_csv(dataset_url)
df.head()

# Normalizing the data frame column names
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(df.columns)

# Fetching all the string columns AKA those with object as pandas dtype
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
print(string_columns)

# Normalizing the values for all the string columns
for column in string_columns:
    df[column] = df[column].str.lower().str.replace(' ', '_')

df[string_columns].head()

# %% Exploratory Data Analysis

for column in df.columns:
    print(f"Name: {column}")
    print(f"Top five unique values: {df[column].unique()[:5]}")
    print(f"Number of unique: {df[column].nunique()}\n")

sns.histplot(df.msrp, bins=50)
plt.show()

sns.histplot(df.msrp[df.msrp < 100_000], bins=50)
plt.show()

# Variable transform: x -> log(x + 1)
price_logs = np.log1p(df.msrp)

# Plotting the log transform
sns.histplot(price_logs, bins=50)
plt.show()

# Missing values
df.isnull().sum().sort_values(ascending=False)

# %% Validation framework

# Total number of records
n = len(df)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - (n_val + n_test)

print(f"Dataframe size: {n}")
print(f"Validation framework size: {n_val + n_test + n_train}")
assert (n == n_val + n_test + n_train)

print(f"Validation size: {n_val}")
print(f"Test size: {n_test}")
print(f"Train size: {n_train}")

df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train + n_val]
df_test = df.iloc[n_train + n_val:]

print(df_train)

# Shuffling the index
idx = np.arange(n)
np.random.seed(2)
np.random.shuffle(idx)

# Redoing the train-test-validation split, with shuffled index
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

# Resetting the indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Fetching the target variable as the underlying numpy ndarray
y_train = np.log1p(df_train['msrp'].values)
y_val = np.log1p(df_val['msrp'].values)
y_test = np.log1p(df_test['msrp'].values)

# Deleting the target from the dataset
del df_train['msrp'], df_val['msrp'], df_test['msrp']

# %% Linear Regression

w0 = 7.17
xi = [453, 11, 86]
w_full = [0.01, 0.04, 0.002]


def linear_regression(x_i):
    """
    Dummy linear regression model with arbitrary weights
    :param x_i: a feature vector
    :return: prediction
    """
    n = len(x_i)
    pred = w0

    for j in range(n):
        pred += w_full[j] * x_i[j]

    return pred


linear_regression(xi)

# Undoing the log(x+1) transform by doing e^(x) - 1
np.expm1(linear_regression(xi))


# %% Linear Regression Vector Form

def dot(xi, w):
    """
    Dot product
    :param xi: vector 1
    :param w: vector 2
    :return: the dot product of both vectors
    """
    n = len(xi)
    res = 0.0
    for j in range(n):
        res += xi[j] * w[j]

    return res


def linear_regression(xi):
    return w0 + dot(xi, w_full)


w_new = [w0] + w_full


def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new)


x1 = [1, 148, 24, 1385]
x2 = [1, 132, 25, 2031]
x10 = [1, 453, 11, 86]

X = [x1, x2, x10]
X = np.array(X)


def linear_regression(X):
    return X.dot(w_new)


# %% Training a linear regression model

def train_linear_regression(X, y):
    ones = np.ones(len(X))
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)

    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]


X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38, 54, 185],
    [142, 25, 431],
    [453, 31, 86],
]
y = [10_000, 20_000, 150_000, 20_050, 10_000, 20_000, 15_000, 25_000, 12_000]
(w0, w) = train_linear_regression(X, y)

# %% Car price baseline model

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
X_train = df_train[base].values

# Missing values
train_linear_regression(X=X_train, y=y_train)

X_train = df_train[base].fillna(0)
w0, w = train_linear_regression(X=X_train, y=y_train)
y_pred = w0 + X_train.dot(w)

sns.histplot(y_pred, color='red', bins=50, alpha=0.5)
sns.histplot(y_train, color='blue', bins=50, alpha=0.5)
plt.show()


# %% Root Mean Square Error

def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)


rmse(y_train, y_pred)

# %% Validating the model

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
X_train = df_train[base].values

# Missing values
train_linear_regression(X=X_train, y=y_train)

X_train = df_train[base].fillna(0)
w0, w = train_linear_regression(X=X_train, y=y_train)
y_pred = w0 + X_train.dot(w)


def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# %% Simple Feature Engineering


# Adding an age variable
def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


# Training and validating the model with the new 'age' variable
X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

# Plotting the predictions vs the target validation var
sns.histplot(y_pred, color='red', bins=50, alpha=0.5)
sns.histplot(y_val, color='blue', bins=50, alpha=0.5)
plt.show()


# %% Categorical variables

def prepare_X(df):
    # Copying the dataframe and creating a new feature named 'age'
    df = df.copy()
    df['age'] = 2017 - df.year

    # Copying the base features and appending the age feature
    features = base.copy()
    features.append('age')

    # Creating dummy variables for the number of doors
    for doors in [2, 3, 4]:
        feature_name = 'num_doors_%s' % doors
        df[feature_name] = (df['number_of_doors'] == doors).astype('int')
        features.append(feature_name)

    car_makes = df_train.make.value_counts().head().index.to_list()
    for make in car_makes:
        feature_name = 'make_%s' % make
        df[feature_name] = (df.make == make).astype('int')
        features.append(feature_name)

    # Filling nulls with 0s and returning the df
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

categorical_variables = ['make', 'engine_fuel_type', 'transmission_type', 'driven_wheels',
                         'market_category', 'vehicle_size', 'vehicle_style']

categories = {}

for c in categorical_variables:
    categories[c] = df[c].value_counts().head().index.to_list()


    def prepare_X(df):
        # Copying the dataframe and creating a new feature named 'age'
        df = df.copy()
        df['age'] = 2017 - df.year

        # Copying the base features and appending the age feature
        features = base.copy()
        features.append('age')

        # Creating dummy variables for the number of doors
        for doors in [2, 3, 4]:
            feature_name = 'num_doors_%s' % doors
            df[feature_name] = (df['number_of_doors'] == doors).astype('int')
            features.append(feature_name)

        for c, values in categories.items():
            for v in values:
                df['%s_%s' % (c, v)] = (df[c] == v).astype('int')
                features.append('%s_%s' % (c, v))

        # Filling nulls with 0s and returning the df
        df_num = df[features]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# %% Regularization

def train_linear_regression_reg(X, y, r=0.01):
    ones = np.ones(len(X))
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)

    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]


X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=0.01)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

# %% Tuning the model

for r in [0.0, 0.00000001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(f"For r={r}, score={score.round(5)}")

# %% Using the model

df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = prepare_X(df_full_train)
y_full_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
rmse(y_test, y_pred)
