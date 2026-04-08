import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.array(x)
    y = np.array(y)

    diff = y -x 
    diff_sq = diff**2
    sum = np.sum(diff_sq)
    dist = np.sqrt(sum)

    return dist

    
    pass