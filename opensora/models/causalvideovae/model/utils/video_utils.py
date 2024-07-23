import torch
import numpy as np

def tensor_to_video(x):
    x = (x * 2 - 1).detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 0, 2, 3).float().numpy() # c t h w -> t c h w
    x = (255 * x).astype(np.uint8)
    return x