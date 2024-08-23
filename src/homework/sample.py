import time
import numpy as np

zeros = np.zeros((10,20)) # a 10x20 Zero matrix, the input to Zeros function is a tuple (#rows, #col)
ones = np.ones((10,20)) # a 10x20 Ones matrix, the input to Ones function is a tuple (#rows, #col)
identity = np.eye(10) # a 10x10 identity matrix
a = np.matrix([[1,2],[5,9]]) # manually creating a matrix
b = np.matrix('1,4;5,8') # second way to manually create a matrix
rand_matrix_a = np.random.random((5,10)) # a random 5x10 matrix
rand_matrix_b = np.random.random((10,2)) # a random 10x2 matrix
col_vect1 = np.array([[1],[-1],[2]])
row_vect1 = np.array([8,1,1])


# print(a)
# print(b)
# print(rand_matrix_a @ rand_matrix_b)
# print(np.multiply(rand_matrix_a,rand_matrix_b))