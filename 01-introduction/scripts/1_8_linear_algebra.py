# Linear Algebra Refresher

import numpy as np

# %% Vector operations

u = np.array([2, 4, 5, 6])
print(u)

v = np.array([1, 0, 0, 2])
print(v)

u + v  # element-wise addition


# %% Vector multiplication

def vector_vector_multiplication(vector_1, vector_2):
    assert vector_1.shape[0] == vector_2.shape[0]
    n = vector_1.shape[0]
    # Using list comprehension
    # result = sum([x * y for (x, y) in zip(vector_1, vector_2)])

    # Using a for loop
    result = 0
    for i in range(n):
        result += vector_1[i] * vector_2[i]

    return result


vector_vector_multiplication(u, v)
u.dot(v)  # The dot product

# %% Matrix-vector multiplication
