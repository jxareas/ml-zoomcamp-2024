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
