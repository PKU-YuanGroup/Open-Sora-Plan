import torch

def nonlinearity(x):
    return x * torch.sigmoid(x)

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)
