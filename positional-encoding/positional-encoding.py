import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Compute sinusoidal positional encodings.
    """
    # 1. Initialize the PE matrix with zeros
    pe = np.zeros((seq_len, d_model))
    
    # 2. Create the column vector of positions: [0, 1, ..., L-1]^T
    # Shape: (seq_len, 1)
    position = np.arange(seq_len).reshape(-1, 1)
    
    # 3. Create the even indices for the denominator
    # We only need one i for every pair (sin, cos)
    i = np.arange(0, d_model, 2)
    
    # 4. Calculate the frequency divisors (The Math you nailed!)
    # We use 2*i because the formula is base^(2i/d_model)
    div_term = base ** (i / d_model)
    
    # 5. Create the 'Angle Matrix' using broadcasting
    # (seq_len, 1) / (1, n_even_indices) -> (seq_len, n_even_indices)
    angles = position / div_term
    
    # 6. Interleave the sine and cosine values
    pe[:, 0::2] = np.sin(angles)             # Even columns
    pe[:, 1::2] = np.cos(angles[:, :d_model//2])  # Odd columns
    
    return pe