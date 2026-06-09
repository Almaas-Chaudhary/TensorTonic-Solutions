import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    Shape: (seq_length, d_model)
    """

    encoding = np.zeros((seq_length, d_model))

    for pos in range(seq_length):
        for dim in range(d_model):
            angle = pos / (10000 ** ((2 * (dim // 2)) / d_model))

            if dim % 2 == 0:
                encoding[pos, dim] = np.sin(angle)
            else:
                encoding[pos, dim] = np.cos(angle)

    return encoding
            