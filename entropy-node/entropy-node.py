import numpy as np

def entropy_node(y):
    # Handle empty nodes
    if np.size(y) == 0:
        return 0.0
    
    # Get counts and calculate probabilities
    _, counts = np.unique(y, return_counts=True)
    probs = counts / np.sum(counts)
    
    # Use only non-zero probabilities for stability
    p_nonzero = probs[probs > 0]
    
    # The final Shannon Entropy formula
    return -np.sum(p_nonzero * np.log2(p_nonzero))