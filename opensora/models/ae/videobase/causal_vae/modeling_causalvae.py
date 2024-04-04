import torch
from torch import nn
from tqdm import tqdm

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
        # self.tile_sample_min_size = self.config.resolution
        self.tile_sample_min_size = 256
        self.tile_sample_min_size_t = 65
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(self.config.ch_mult) - 1)))
        self.tile_latent_min_size_t = int((self.tile_sample_min_size_t-1) / (2 ** self.config.time_compress)) + 1
        self.tile_overlap_factor = 0.25
        self.use_tiling = False

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
        # print(self.use_tiling, x.shape, self.tile_sample_min_size, self.tile_latent_min_size, self.tile_sample_min_size_t, self.tile_latent_min_size_t)
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            # print('tileing')
            return self.tiled_encode2d(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            # print('tileing')
            return self.tiled_decode2d(z)
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



    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode2d(self, x):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    def tiled_decode2d(self, z):

        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec


    # def tiled_encode(self, x):
    #     overlap_size = max(int(self.tile_sample_min_size * (1 - self.tile_overlap_factor)), 1)
    #     blend_extent = max(int(self.tile_latent_min_size * self.tile_overlap_factor), 1)
    #     row_limit = self.tile_latent_min_size - blend_extent
    #
    #     overlap_size_t = max(int((self.tile_sample_min_size_t - 1) * (1 - self.tile_overlap_factor)) + 1, 1)
    #     blend_extent_t = max(int((self.tile_latent_min_size_t - 1) * self.tile_overlap_factor) + 1, 1)
    #     row_limit_t = self.tile_latent_min_size_t - blend_extent_t
    #
    #     # Split the image into 512x512 tiles and encode them separately.
    #     rows_t = []
    #     # import ipdb
    #     # ipdb.set_trace()
    #     for t in tqdm(range(0, x.shape[2], overlap_size_t)):
    #         rows = []
    #         for i in range(0, x.shape[3], overlap_size):
    #             row = []
    #             for j in range(0, x.shape[4], overlap_size):
    #                 tile = x[:, :, t: t + self.tile_sample_min_size_t, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
    #                 tile = self.encoder(tile)
    #                 tile = self.quant_conv(tile)
    #                 row.append(tile)
    #             rows.append(row)
    #         rows_t.append(rows)
    #
    #     result_rows_t = []
    #     for t, rows in enumerate(rows_t):
    #         result_rows = []
    #         for i, row in enumerate(rows):
    #             result_row = []
    #             for j, tile in enumerate(row):
    #                 # blend the above tile and the left tile
    #                 # to the current tile and add the current tile to the result row
    #                 if t > 0:
    #                     tile = self.blend_t(rows_t[t - 1][i][j], tile, blend_extent_t)
    #                 if i > 0:
    #                     tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
    #                 if j > 0:
    #                     tile = self.blend_h(row[j - 1], tile, blend_extent)
    #                 result_row.append(tile[:, :, :row_limit_t, :row_limit, :row_limit])
    #             result_rows.append(torch.cat(result_row, dim=4))
    #         result_rows_t.append(torch.cat(result_rows, dim=3))
    #
    #     moments = torch.cat(result_rows_t, dim=2)
    #     posterior = DiagonalGaussianDistribution(moments)
    #
    #     return posterior
    #
    # def tiled_decode(self, z):
    #     overlap_size = max(int(self.tile_latent_min_size * (1 - self.tile_overlap_factor)), 1)
    #     blend_extent = max(int(self.tile_sample_min_size * self.tile_overlap_factor), 1)
    #     row_limit = self.tile_sample_min_size - blend_extent
    #
    #     overlap_size_t = max(int((self.tile_latent_min_size_t - 1) * (1 - self.tile_overlap_factor)) + 1, 1)
    #     blend_extent_t = max(int((self.tile_sample_min_size_t - 1) * self.tile_overlap_factor) + 1, 1)
    #     row_limit_t = self.tile_sample_min_size_t - blend_extent_t
    #
    #     # Split z into overlapping 64x64 tiles and decode them separately.
    #     # The tiles have an overlap to avoid seams between tiles.
    #     rows_t = []
    #     for t in tqdm(range(0, z.shape[2], overlap_size_t)):
    #         rows = []
    #         for i in range(0, z.shape[3], overlap_size):
    #             row = []
    #             for j in range(0, z.shape[4], overlap_size):
    #                 tile = z[:, :, t: t + self.tile_latent_min_size_t, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
    #                 tile = self.post_quant_conv(tile)
    #                 decoded = self.decoder(tile)
    #                 row.append(decoded)
    #             rows.append(row)
    #         rows_t.append(rows)
    #
    #     result_rows_t = []
    #     for t, rows in enumerate(rows_t):
    #         result_rows = []
    #         for i, row in enumerate(rows):
    #             result_row = []
    #             for j, tile in enumerate(row):
    #                 # blend the above tile and the left tile
    #                 # to the current tile and add the current tile to the result row
    #                 if t > 0:
    #                     tile = self.blend_t(rows_t[t - 1][i][j], tile, blend_extent_t)
    #                 if i > 0:
    #                     tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
    #                 if j > 0:
    #                     tile = self.blend_h(row[j - 1], tile, blend_extent)
    #                 result_row.append(tile[:, :, :row_limit_t, :row_limit, :row_limit])
    #             result_rows.append(torch.cat(result_row, dim=4))
    #         result_rows_t.append(torch.cat(result_rows, dim=3))
    #
    #     dec = torch.cat(result_rows_t, dim=2)
    #     return dec
    #
    # def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent_t: int) -> torch.Tensor:
    #     blend_extent_t = min(a.shape[2], b.shape[2], blend_extent_t)
    #     for t in range(blend_extent_t):
    #         b[:, :, t, :, :] = a[:, :, -blend_extent_t + t, :, :] * (1 - t / blend_extent_t) + b[:, :, t, :, :] * (
    #                     t / blend_extent_t)
    #     return b
    #
    # def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    #     blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    #     for y in range(blend_extent):
    #         b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
    #                     y / blend_extent)
    #     return b
    #
    # def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    #     blend_extent = min(a.shape[4], b.shape[4], blend_extent)
    #     for x in range(blend_extent):
    #         b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
    #                     x / blend_extent)
    #     return b

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)