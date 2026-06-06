import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    def posi(pos, dim):
        pp = pos / (base ** ((2 * (dim // 2)) / d_model))

        if dim % 2 == 0:
            pe = np.sin(pp)
        else:
            pe = np.cos(pp)

        return pe

    encoding = [[0] * d_model for _ in range(seq_len)]

    for pos in range(seq_len):
        for dim in range(d_model):
            encoding[pos][dim] = posi(pos, dim)

    return np.array(encoding)
    