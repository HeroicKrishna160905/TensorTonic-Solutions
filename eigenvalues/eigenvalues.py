import numpy as np

def calculate_eigenvalues(matrix):
    try:
        # 1. Convert to a numeric array
        matrix = np.array(matrix, dtype=float)
    except:
        # Catches jagged arrays or non-numeric input
        return None

    # 2. The Universal Square Check
    # A square matrix MUST be 2D (ndim == 2) 
    # AND it must have equal rows and columns.
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    
    # 3. Handle the case of an actually empty 2D matrix (like [[]])
    # if the requirements allow it, otherwise the line above handles it.
    if matrix.size == 0:
        return np.array([])

    # 4. Calculate and Sort
    eigenvalues = np.linalg.eigvals(matrix)
    return np.sort_complex(eigenvalues)