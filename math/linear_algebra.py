import numpy as np

def vector_sum(v1, v2):
    return np.add(v1, v2)

def dot_product(v1, v2):
    return np.dot(v1, v2)

def vector_norm(v):
    return np.linalg.norm(v)

def matrix_multiply(A, B):
    return np.dot(A, B)
