import numpy as np

def ridge_regression(X, y, lam):
    # Convert inputs to numpy arrays for linear algebra operations
    X = np.array(X)
    y = np.array(y)
    
    # 1. Identify the number of features (n)
    n_features = X.shape[1]
    
    # 2. Calculate A = (X^T @ X + lambda * I)
    # np.eye(n) creates the n x n Identity matrix
    A = X.T @ X + lam * np.eye(n_features)
    
    # 3. Calculate w = A^-1 @ X^T @ y
    w = np.linalg.inv(A) @ X.T @ y
    
    # Return the weights as a list of floats
    return w.tolist()