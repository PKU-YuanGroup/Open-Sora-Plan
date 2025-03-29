import math
from einops import rearrange
import decord
from torch.nn import functional as F
import torch


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str

def pad_to_multiple(number, ds_stride):
    remainder = number % ds_stride
    if remainder == 0:
        return number
    else:
        padding = ds_stride - remainder
        return number + padding
