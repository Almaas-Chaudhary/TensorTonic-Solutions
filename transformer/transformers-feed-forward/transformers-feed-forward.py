import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    first = np.dot(x, W1) + b1
    maxx = np.maximum(0, first)
    output = np.dot(maxx, W2) + b2    
    return output