import math
from torch.nn import functional as F
import torch


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
        self.vae_stride = args.vae_stride
        self.patch_size = args.patch_size
        self.patch_size_t = args.patch_size_t
        self.num_frames = args.num_frames

    def __call__(self, batch):
        batch_tubes, labels = tuple(zip(*batch))
        labels = torch.as_tensor(labels).to(torch.long)
        ds_stride = self.vae_stride * self.patch_size
        t_ds_stride = self.vae_stride * self.patch_size_t

        # pad to max multiple of ds_stride
        batch_input_size = [i.shape for i in batch_tubes]
        max_t, max_h, max_w = max([i[1] for i in batch_input_size]), \
                              max([i[2] for i in batch_input_size]), \
                              max([i[3] for i in batch_input_size])
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
        max_latent_size = [max_tube_size[0] // self.vae_stride,
                           max_tube_size[1] // self.vae_stride,
                           max_tube_size[2] // self.vae_stride]
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

        return pad_batch_tubes, labels, attention_mask

