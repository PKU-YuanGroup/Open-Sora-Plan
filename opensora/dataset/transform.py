import math
import random

import torch
from torchvision import transforms
from torchvision.transforms import Lambda



class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index


class LongSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(
            self, size: int, interpolation: str = "bilinear"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert len(x.shape) == 4
        assert x.dtype == torch.float32
        c, t, h, w = x.shape
        if w < h:
            new_w = int(math.floor((float(w) / h) * self._size))
            new_h = self._size
        else:
            new_w = self._size
            new_h = int(math.floor((float(h) / w) * self._size))
        return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=self._interpolation, align_corners=False)
