import numpy as np

M = np.array([[1,0],[0,1]])

# Eigen value Decomposition
eigvals, eigvecs = np.linalg.eig(M)

# Single Value Decomposition

# Example:
A = np.ones((100,200))
