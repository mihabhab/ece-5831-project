import time
import numpy as np

# FUNCTION DEFINITIONS
def list_dot_product(a:float, b:float):
    return sum(x * y for x, y in zip(a, b))

def list_matrix_vector_multiply(matrix, vector):
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]

def list_matrix_inverse(matrix): # Expects a 2x2 matrix
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    inv_det = 1 / det
    return [
        [matrix[1][1] * inv_det, -matrix[0][1] * inv_det],
        [-matrix[1][0] * inv_det, matrix[0][0] * inv_det]
    ]

# Python lists implementation
X_list = [[2, 4], [1, 3]]
y_list = [1, 3]
z_list = [2, 3]

yTz_list = list_dot_product(y_list, z_list)
Xy_list = list_matrix_vector_multiply(X_list, y_list)
X_inv_list = list_matrix_inverse(X_list)

# NumPy arrays implementation
X_np = np.array([[2, 4], [1, 3]])
y_np = np.array([1, 3])
z_np = np.array([2, 3])

yTz_np = np.dot(y_np, z_np)
Xy_np = np.dot(X_np, y_np)
X_inv_np = np.linalg.inv(X_np)

# checking that both methods yield the same values
print(f'yTz: Python List: {yTz_list}, numpy array: {yTz_np}')
print(f'Xy: Python List: {Xy_list}, numpy array: {Xy_np}')
print(f'X^-1: Python List: {X_inv_list}, numpy array: {X_inv_np}')

# ITERATION TIMING: PYTHON LIST IMPLEMENTATION
start_time = time.time() # Starting timer before the loop

for _ in range(10000):
    yTz_list = list_dot_product(y_list, z_list)
    Xy_list = list_matrix_vector_multiply(X_list, y_list)
    X_inv_list = list_matrix_inverse(X_list)

end_time = time.time()
list_time = end_time - start_time

print(f"Python lists implementation time: {list_time} seconds")

# ITERATION TIMING: NUMPY ARRAY IMPLEMENTATION
start_time = time.time()

for _ in range(10000):
    yTz_np = np.dot(y_np, z_np)
    Xy_np = np.dot(X_np, y_np)
    X_inv_np = np.linalg.inv(X_np)

end_time = time.time()
numpy_time = end_time - start_time

print(f"NumPy arrays implementation time: {numpy_time} seconds")

# TIMING RATIO
time_ratio = numpy_time/list_time
print(f'numpy array to Python list performance has a ratio of {time_ratio}')

# Numpy vs. Python List
# 2x2 matrix Python list outperforms numpy
# 10x10 numpy outperforms Python list