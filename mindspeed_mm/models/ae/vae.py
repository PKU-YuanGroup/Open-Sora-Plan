import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from einops import rearrange
from megatron.core import mpu
from mindspeed_mm.models.common.communications import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from .vae_temporal import VAE_Temporal


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


class VideoAutoencoder3D(nn.Module):
    def __init__(self, config, micro_frame_size=17):
        super().__init__()
        self.spatial_vae = VideoAutoencoderKL(config, micro_batch_size=4)
        self.temporal_vae = VAE_Temporal()
        self.cal_loss = config.cal_loss
        self.micro_frame_size = config.micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size([config.micro_frame_size, None, None])[0]

        if config.freeze_vae_2d:
            for param in self.spatial_vae.parameters():
                param.requires_grad = False

        self.out_channels = self.temporal_vae.out_channels

        # normalization parameters
        scale = torch.tensor(config.scale)
        shift = torch.tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def encode(self, x):
        x_z = self.spatial_vae.encode(x)

        if self.micro_frame_size is None:
            posterior = self.temporal_vae.encode(x_z)
            z = posterior.sample()
        else:
            z_list = []
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i : i + self.micro_frame_size]
                posterior = self.temporal_vae.encode(x_z_bs)
                z_list.append(posterior.sample())
            z = torch.cat(z_list, dim=2)

        if self.cal_loss:
            return z, posterior, x_z
        else:
            return (z - self.shift) / self.scale

    def decode(self, z, num_frames=None):
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z)
        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i : i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)
            x = self.spatial_vae.decode(x_z)

        if self.cal_loss:
            return x, x_z
        else:
            return x

    def forward(self, x):
        if not self.cal_loss:
            raise Exception("This method is only available when cal_loss is True")
        z, posterior, x_z = self.encode(x)
        x_rec, x_z_rec = self.decode(z, num_frames=x_z.shape[2])
        return x_rec, x_z_rec, z, posterior, x_z

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))
        else:
            sub_input_size = [self.micro_frame_size, input_size[1], input_size[2]]
            sub_latent_size = self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(sub_input_size))
            sub_latent_size[0] = sub_latent_size[0] * (input_size[0] // self.micro_frame_size)
            remain_temporal_size = [input_size[0] % self.micro_frame_size, None, None]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(remain_temporal_size)
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size

    def get_temporal_last_layer(self):
        return self.temporal_vae.decoder.conv_out.conv.weight

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype