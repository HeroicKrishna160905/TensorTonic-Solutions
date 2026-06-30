import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    try:
        M = np.asarray(matrix, dtype=float)

        if M.ndim != 2:
            return None

        if norm_type == "l2":
            norm = np.sqrt(np.sum(M**2, axis=axis, keepdims=True))
        elif norm_type == "l1":
            norm = np.sum(np.abs(M), axis=axis, keepdims=True)
        elif norm_type == "max":
            norm = np.max(np.abs(M), axis=axis, keepdims=True)
        else:
            return None

        # Avoid division by zero
        norm[norm == 0] = 1

        return M / norm

    except:
        return None