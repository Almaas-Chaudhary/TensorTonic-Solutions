# import numpy as np

# def dropout(x, p=0.5, rng=None):

#     x = np.array(x, dtype=float)

#     if rng is not None:
#         mask = (rng.random(x.shape) > p).astype(float)
#     else:
#         mask = (np.random.random(x.shape) > p).astype(float)

#     pattern = mask / (1 - p)

#     output = x * pattern

#     return output, pattern
import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x, dtype=float)

    # Choose random number generator
    if rng is not None:
        random = rng.random 
    else:
        np.random.random

    # Keep neurons with probability (1 - p)
    keep_mask = random(x.shape) > p

    # Inverted dropout scaling
    pattern = keep_mask.astype(float) / (1 - p)

    # Apply dropout
    output = x * pattern

    return output, pattern