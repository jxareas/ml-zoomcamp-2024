## Homework

# %% Set up the environment

# You need to install Python, NumPy, Pandas, Matplotlib and Seaborn.
# For that, you can use the instructions from:
# [06-environment.md](../../../01-intro/06-environment.md)
import pandas as pd
import numpy as np

# %% Q1. Pandas version
# What's the version of Pandas that you installed?
# You can get the version information using the `__version__` field:
# print(pd.__version__)

print(pd.__version__)  # 2.2.2

# %% Getting the data
# For this hw, we'll use the Laptops Price dataset. Download it from
# https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv
# You can do it with wget:
# !wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv
# Or just open it with your browser and click "Save as...".
# Now read it with Pandas.

df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")

# %% Q2. Records count
# How many records are in the dataset?
# - 12
# - 1000
# - 2160
# - 12160
len(df)  # 2'160

# %% Q3. Laptop brands
# How many laptop brands are presented in the dataset?
# - 12
# - 27
# - 28
# - 2160
df['Brand'].nunique()  # 27

# %% Q4. Missing values
# How many columns in the dataset have missing values?
# - 0
# - 1
# - 2
# - 3
has_at_least_one_missing_value = df.isnull().sum() > 0
missing_columns = df.columns[has_at_least_one_missing_value]
print(f"{missing_columns.values}: {len(missing_columns)}")  # 3

# %% Q5. Maximum final price
# What's the maximum final price of Dell notebooks in the dataset?
# - 869
# - 3691
# - 3849
# - 3936
max_final_price_dell_notebooks = df[df['Brand'] == 'Dell']['Final Price'].max()
print(f"Maximum final price of dell notebooks is {max_final_price_dell_notebooks}")  # 3936

# %% Q6. Median value of Screen
# 1. Find the median value of `Screen` column in the dataset.
median_screen = df['Screen'].median()
# 2. Next, calculate the most frequent value of the same `Screen` column.
most_frequent_screen_value = df['Screen'].mode()[0]
# 3. Use `fillna` method to fill the missing values in `Screen` column with the most frequent value from the previous step.
df['Screen'] = df['Screen'].fillna(most_frequent_screen_value)
# 4. Now, calculate the median value of `Screen` once again.
df['Screen'].median()

# Has it changed?
# - Yes
# - No
print(f"Did the screen median value change? ==> {df['Screen'].median() != median_screen}")  # No

# %% Q7. Sum of weights
# 1. Select all the "Innjoo" laptops from the dataset.
innjoo_laptops = df[df['Brand'] == 'Innjoo']
# 2. Select only columns `RAM`, `Storage`, `Screen`.
innjoo_laptops_subset = innjoo_laptops[['RAM', 'Storage', 'Screen']]
# 3. Get the underlying NumPy array. Let's call it `X`.
X = innjoo_laptops_subset.values
# 4. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.
XTX = X.T.dot(X)
# 5. Compute the inverse of `XTX`.
XTX_inv = np.linalg.inv(XTX)
# 6. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100]`.
y = np.array([1100, 1300, 800, 900, 1000, 1100])
# 7. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.
w = XTX_inv.dot(X.T).dot(y)
# 8. What's the sum of all the elements of the result?
# - 0.43
# - 45.29
# - 45.58
# - 91.30
print(f"Sum of all the elements of the result: {sum(w)}")  # 91.299 ~ 91.30

# Submit the results
# Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2024/homework/hw01
# If your answer doesn't match options exactly, select the closest one
