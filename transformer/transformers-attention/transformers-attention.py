import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])

    weights = F.softmax(scores, dim=-1)

    attention = torch.matmul(weights, V)
    return attention