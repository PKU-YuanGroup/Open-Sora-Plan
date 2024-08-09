import torch

def cosine_scheduler(step, max_steps, value_base=1, value_end=0):
    step = torch.tensor(step)
    cosine_value = 0.5 * (1 + torch.cos(torch.pi * step / max_steps))
    value = value_end + (value_base - value_end) * cosine_value
    return value