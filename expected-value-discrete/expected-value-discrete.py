import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
  
    x = np.asarray(x)
    p = np.asarray(p)

   
    if x.shape != p.shape:
        raise ValueError("The shapes of values (x) and probabilities (p) must match.")

    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1.")

    # compute E[X] = sum(x * p)
    expected_val = np.dot(x, p)
    
    return float(expected_val)
