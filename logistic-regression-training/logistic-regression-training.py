import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N, D = X.shape
    
    # 1. Initialize parameters
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # 2. Forward pass: compute predictions
        # z = Xw + b (dot product for features, broadcasting for bias)
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 3. Compute gradients
        # The gradient of BCE w.r.t. the linear output (z) is (p - y)
        error = p - y
        dw = (1 / N) * np.dot(X.T, error)
        db = (1 / N) * np.sum(error)
        
        # 4. Update parameters
        w -= lr * dw
        b -= lr * db
        
    return w, b