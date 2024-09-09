import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from einops import rearrange
from megatron.core import mpu
from mindspeed_mm.models.common.communications import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)


class VideoAutoencoderKL(nn.Module):

    def __init__(
        self, 
        from_pretrained, 
        micro_batch_size=None, 
        patch_size=(1, 8, 8),
        enable_sequence_parallelism=False,
        **kwargs,
    ):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(from_pretrained)
        self.out_channels = self.module.config.latent_channels
        self.micro_batch_size = micro_batch_size
        self.patch_size = patch_size
        self.sp_size = mpu.get_context_parallel_world_size()
        self.enable_sequence_parallelism = (
            enable_sequence_parallelism and self.sp_size > 1 and self.patch_size[0] == 1
        )

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        T = x.shape[2]
        if self.enable_sequence_parallelism and T % self.sp_size == 0:
            x = split_forward_gather_backward(
                x, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
            )
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul(0.18215)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        if self.enable_sequence_parallelism and T % self.sp_size == 0:
            x = gather_forward_split_backward(
                x, mpu.get_context_parallel_group(), dim=2, grad_scale="up"
            )
        return x

    def decode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / 0.18215).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        num_dim = len(self.patch_size)
        for i in range(num_dim):
            if input_size[i] % self.patch_size[i] != 0:
                raise AssertionError("Input size must be divisible by patch size")
        input_size = [input_size[i] // self.patch_size[i] for i in range(num_dim)]
        return input_size
