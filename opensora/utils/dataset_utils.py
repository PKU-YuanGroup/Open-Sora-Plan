import math

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

class Collate:
    def __init__(self, args):
        self.max_image_size = args.max_image_size
        self.ae_stride = args.ae_stride
        self.ae_stride_t = args.ae_stride_t
        self.patch_size = args.patch_size
        self.patch_size_t = args.patch_size_t
        self.num_frames = args.num_frames

    def __call__(self, batch):
        unzip = tuple(zip(*batch))
        if len(unzip) == 2:
            batch_tubes, labels = unzip
            labels = torch.as_tensor(labels).to(torch.long)
        elif len(unzip) == 3:
            batch_tubes, input_ids, cond_mask = unzip
            input_ids = torch.stack(input_ids).squeeze(1)
            cond_mask = torch.stack(cond_mask).squeeze(1)
        else:
            raise NotImplementedError
        ds_stride = self.ae_stride * self.patch_size
        t_ds_stride = self.ae_stride_t * self.patch_size_t

        # pad to max multiple of ds_stride
        batch_input_size = [i.shape for i in batch_tubes]
        max_t, max_h, max_w = self.num_frames, \
                              self.max_image_size, \
                              self.max_image_size
        pad_max_t, pad_max_h, pad_max_w = pad_to_multiple(max_t, t_ds_stride), \
                                          pad_to_multiple(max_h, ds_stride), \
                                          pad_to_multiple(max_w, ds_stride)
        each_pad_t_h_w = [[pad_max_t - i.shape[1],
                           pad_max_h - i.shape[2],
                           pad_max_w - i.shape[3]] for i in batch_tubes]
        pad_batch_tubes = [F.pad(im,
                                 (0, pad_w,
                                  0, pad_h,
                                  0, pad_t), value=0) for (pad_t, pad_h, pad_w), im in zip(each_pad_t_h_w, batch_tubes)]
        pad_batch_tubes = torch.stack(pad_batch_tubes, dim=0)

        # make attention_mask
        max_tube_size = [pad_max_t, pad_max_h, pad_max_w]
        max_latent_size = [max_tube_size[0] // self.ae_stride_t,
                           max_tube_size[1] // self.ae_stride,
                           max_tube_size[2] // self.ae_stride]
        max_patchify_latent_size = [max_latent_size[0] // self.patch_size_t,
                                    max_latent_size[1] // self.patch_size,
                                    max_latent_size[2] // self.patch_size]
        valid_patchify_latent_size = [[int(math.ceil(i[1] / t_ds_stride)),
                                       int(math.ceil(i[2] / ds_stride)),
                                       int(math.ceil(i[3] / ds_stride))] for i in batch_input_size]
        attention_mask = [F.pad(torch.ones(i),
                                (0, max_patchify_latent_size[2] - i[2],
                                 0, max_patchify_latent_size[1] - i[1],
                                 0, max_patchify_latent_size[0] - i[0]), value=0) for i in valid_patchify_latent_size]
        attention_mask = torch.stack(attention_mask)

        if len(unzip) == 2:
            return pad_batch_tubes, labels, attention_mask
        elif len(unzip) == 3:
            return pad_batch_tubes, attention_mask, input_ids, cond_mask

