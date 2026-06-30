import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    try:
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[0] < 2:
            return None

        # Center the data
        Xc = X - np.mean(X, axis=0)

        # Covariance matrix
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)

        # Standard deviations
        std = np.sqrt(np.diag(cov))

        # Outer product of std deviations
        denom = np.outer(std, std)

        # Correlation matrix
        corr = cov / denom

        # Handle zero-variance features
        corr[denom == 0] = np.nan

        return corr

    except:
        return None