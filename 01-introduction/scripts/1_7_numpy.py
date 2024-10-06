# Introduction to Numpy
#
# Plan:
#
# * Creating arrays
# * Multi-dimensional arrays
# * Randomly generated arrays
# * Element-wise operations
#     * Comparison operations
#     * Logical operations
# * Summarizing operations

# %% Importing numpy
import numpy as np

np

# %% Creating arrays

# Creating an array of zeros with 10 elements
np.zeros(10)

# Creating an array of ones with 10 elements
np.ones(10)

# Creating an array of 2.5s with 10 elements
np.full(shape=10, fill_value=2.5)

# Converting a list to numpy arrays
a = np.array([1, 2, 3, 5, 7, 12])

# Indexing
print(a[2])  # Accessing third element, with a value of 3

a[2] = 10  # Setting the value of the third element to 3
print(a[2])  # Accessing third element, now with a value of 10

# Creating an array with numbers from 0 to 9
np.arange(10)

np.linspace(start=0, stop=1, num=11)

# %% Multidimensional Arrays

# Creating an array of 5 rows and 2 columns
np.zeros((5, 2))

# Converting a list of lists to a numpy ndarray
n = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

# Grabbing the 7: third row and first column
print(n[2, 0])

# Replacing 2 with 20
n[0, 1] = 20

# Outputs 20
print(n[0, 1])

# Accessing an entire row
print(n[1])

# Rewriting a row
n[2] = np.ones(3)
print(n)

# Accessing an entire column
print(n[:, 1])

# Rewriting a column
n[:, 2] = [0, 1, 2]
print(n)

# %% Randomly generated arrays

np.random.seed(2)
np.random.rand(5, 2)  # Standard uniform distribution, random numbers between 0 and 1

# %% Probability distributions

np.random.seed(2)
np.random.randn(5, 2)  # Standard normal distribution

np.random.seed(2)
100 * np.random.rand(5, 2)  # Pseudo-random numbers between 0 and 100

np.random.seed(2)
np.random.randint(low=0, high=100, size=(5, 2))  # Random integers between 0 and 99

# %% Element-wise operations

a = np.arange(5)

# Adding 1 to each element of the array
a + 1

# Multiplying 2 to each element of the array
a * 2

# Chaining operations
b = (10 + (a * 2)) ** 2 / 100

# Combining to arrays
a + b

# %% Element-wise comparison operations

print(a)
print(a >= 2)

print(a > b)

# All elements in a which are greater than b's by element-wise comparison
print(a[a > b])

# %% Summarizing operations

# Minimum
a.min()

# Maximum
a.max()

# Mean
a.mean()

# Standard deviation
a.std()
