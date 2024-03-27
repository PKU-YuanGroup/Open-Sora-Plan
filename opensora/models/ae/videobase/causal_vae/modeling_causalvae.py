import torch
from torch import nn
from .configuration_causalvae import CausalVAEConfiguration
from ..modules.attention import make_attn
from ..modules.resnet_block import ResnetBlock3D
from ..modules.conv import CausalConv3d
from ..modules.normalize import Normalize
from ..modules.perceptual_loss import LPIPSWithDiscriminator
from ..modules.updownsample import (
    SpatialDownsample2x,
    SpatialUpsample2x,
    TimeDownsample2x,
    TimeUpsample2x,
)
import numpy as np
from ..modules.ops import nonlinearity
from ..modeling_videobase import VideoBaseAE

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        time_compress=2,
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.time_compress = time_compress
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = CausalConv3d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = SpatialDownsample2x(block_in, block_in)
                curr_res = curr_res // 2
            # if i_level < self.time_compress and i_level != self.num_resolutions - 1:
            if i_level < self.time_compress:
                down.time_downsample = TimeDownsample2x()
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CausalConv3d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            # if i_level < self.time_compress and i_level != self.num_resolutions - 1:
            if i_level < self.time_compress:
                hs.append(self.down[i_level].time_downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        time_compress=2,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.time_compress = time_compress
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = SpatialUpsample2x(block_in, block_in)
                curr_res = curr_res * 2
            if i_level > self.num_resolutions - 1 - self.time_compress and i_level != 0:
            # if i_level <= self.time_compress and i_level != 0:
                up.time_upsample = TimeUpsample2x()

            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        # z to block_in
        h = self.conv_in(z)
        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            
            if i_level > self.num_resolutions - 1 - self.time_compress and i_level != 0:
            # if i_level <= self.time_compress and i_level != 0:
                h = self.up[i_level].time_upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class CausalVAEModel(VideoBaseAE):
    CONFIGURATION_CLS = CausalVAEConfiguration

    def __init__(self, config: CausalVAEConfiguration):
        super().__init__()
        self.config = config
        self.loss = LPIPSWithDiscriminator(
            logvar_init=config.logvar_init,
            kl_weight=config.kl_weight,
            pixelloss_weight=config.pixelloss_weight,
            perceptual_weight=config.perceptual_weight,
            disc_loss=config.disc_loss,
        )
        self.loss.logvar.requires_grad = False
        self.encoder = Encoder(
            ch=config.hidden_size,
            out_ch=config.out_channels,
            ch_mult=config.ch_mult,
            z_channels=config.z_channels,
            num_res_blocks=config.num_res_block,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            in_channels=config.in_channels,
            resolution=config.resolution,
            resamp_with_conv=True,
            attn_type=config.attn_type,
            use_linear_attn=config.use_linear_attn,
            time_compress=config.time_compress,
        )
        self.decoder = Decoder(
            ch=config.hidden_size,
            out_ch=config.out_channels,
            ch_mult=config.ch_mult,
            z_channels=config.z_channels,
            num_res_blocks=config.num_res_block,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            in_channels=config.in_channels,
            resolution=config.resolution,
            resamp_with_conv=True,
            attn_type=config.attn_type,
            use_linear_attn=config.use_linear_attn,
            time_compress=config.time_compress,
        )
        self.quant_conv = CausalConv3d(
            2 * config.z_channels, 2 * config.embed_dim, 1
        )
        self.post_quant_conv = CausalConv3d(
            config.embed_dim, config.z_channels, 1
        )
        self.embed_dim = config.embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")