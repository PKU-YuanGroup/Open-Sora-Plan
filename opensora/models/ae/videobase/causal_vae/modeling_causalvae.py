try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
from ..modeling_videobase import VideoBaseAE_PL
from ..modules import Normalize
from ..modules.ops import nonlinearity
from typing import List, Tuple
import torch.nn as nn
from ..utils.module_utils import resolve_str_to_obj, Module
from ..utils.distrib_utils import DiagonalGaussianDistribution
from ..utils.scheduler_utils import cosine_scheduler
import torch
from diffusers.configuration_utils import register_to_config


class Encoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: Module = "Conv2d",
        conv_out: Module = "CasualConv3d",
        attention: Module = "AttnBlock",
        resnet_blocks: Tuple[Module] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock3D",
        ),
        spatial_downsample: Tuple[Module] = (
            "Downsample",
            "Downsample",
            "Downsample",
            "",
        ),
        temporal_downsample: Tuple[Module] = ("", "", "TimeDownsampleRes2x", ""),
        mid_resnet: Module = "ResnetBlock3D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        assert len(resnet_blocks) == len(hidden_size_mult), print(
            hidden_size_mult, resnet_blocks
        )
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        # ---- In ----
        self.conv_in = resolve_str_to_obj(conv_in)(
            3, hidden_size, kernel_size=3, stride=1, padding=1
        )

        # ---- Downsample ----
        curr_res = resolution
        in_ch_mult = (1,) + tuple(hidden_size_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_size * in_ch_mult[i_level]
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    resolve_str_to_obj(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_obj(attention)(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if spatial_downsample[i_level]:
                down.downsample = resolve_str_to_obj(spatial_downsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res // 2
            if temporal_downsample[i_level]:
                down.time_downsample = resolve_str_to_obj(temporal_downsample[i_level])(
                    block_in, block_in
                )
            self.down.append(down)

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = resolve_str_to_obj(attention)(block_in)
        self.mid.block_2 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        # ---- Out ----
        self.norm_out = Normalize(block_in)
        self.conv_out = resolve_str_to_obj(conv_out)(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))
            if hasattr(self.down[i_level], "time_downsample"):
                hs_down = self.down[i_level].time_downsample(hs[-1])
                hs.append(hs_down)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        if npu_config is None:
            h = self.norm_out(h)
        else:
            h = npu_config.run_group_norm(self.norm_out, h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: Module = "Conv2d",
        conv_out: Module = "CasualConv3d",
        attention: Module = "AttnBlock",
        resnet_blocks: Tuple[Module] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        spatial_upsample: Tuple[Module] = (
            "",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
        ),
        temporal_upsample: Tuple[Module] = ("", "", "", "TimeUpsampleRes2x"),
        mid_resnet: Module = "ResnetBlock3D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        # ---- In ----
        block_in = hidden_size * hidden_size_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = resolve_str_to_obj(conv_in)(
            z_channels, block_in, kernel_size=3, padding=1
        )

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = resolve_str_to_obj(attention)(block_in)
        self.mid.block_2 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # ---- Upsample ----
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resolve_str_to_obj(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_obj(attention)(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if spatial_upsample[i_level]:
                up.upsample = resolve_str_to_obj(spatial_upsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res * 2
            if temporal_upsample[i_level]:
                up.time_upsample = resolve_str_to_obj(temporal_upsample[i_level])(
                    block_in, block_in
                )
            self.up.insert(0, up)

        # ---- Out ----
        self.norm_out = Normalize(block_in)
        self.conv_out = resolve_str_to_obj(conv_out)(
            block_in, 3, kernel_size=3, padding=1
        )

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
            if hasattr(self.up[i_level], "time_upsample"):
                h = self.up[i_level].time_upsample(h)
        if npu_config is None:
            h = self.norm_out(h)
        else:
            h = npu_config.run_group_norm(self.norm_out, h)
        h = nonlinearity(h)
        if npu_config is None:
            h = self.conv_out(h)
        else:
            h_dtype = h.dtype
            h = npu_config.run_conv3d(self.conv_out, h, h_dtype)
        return h


class CausalVAEModel(VideoBaseAE_PL):

    @register_to_config
    def __init__(
        self,
        lr: float = 1e-5,
        hidden_size: int = 128,
        z_channels: int = 4,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = [],
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        loss_type: str = "opensora.models.ae.videobase.losses.LPIPSWithDiscriminator",
        loss_params: dict = {
            "kl_weight": 0.000001,
            "logvar_init": 0.0,
            "disc_start": 2001,
            "disc_weight": 0.5,
        },
        q_conv: str = "CausalConv3d",
        encoder_conv_in: Module = "CausalConv3d",
        encoder_conv_out: Module = "CausalConv3d",
        encoder_attention: Module = "AttnBlock3D",
        encoder_resnet_blocks: Tuple[Module] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        encoder_spatial_downsample: Tuple[Module] = (
            "SpatialDownsample2x",
            "SpatialDownsample2x",
            "SpatialDownsample2x",
            "",
        ),
        encoder_temporal_downsample: Tuple[Module] = (
            "",
            "TimeDownsample2x",
            "TimeDownsample2x",
            "",
        ),
        encoder_mid_resnet: Module = "ResnetBlock3D",
        decoder_conv_in: Module = "CausalConv3d",
        decoder_conv_out: Module = "CausalConv3d",
        decoder_attention: Module = "AttnBlock3D",
        decoder_resnet_blocks: Tuple[Module] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        decoder_spatial_upsample: Tuple[Module] = (
            "",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
        ),
        decoder_temporal_upsample: Tuple[Module] = ("", "", "TimeUpsample2x", "TimeUpsample2x"),
        decoder_mid_resnet: Module = "ResnetBlock3D",
    ) -> None:
        super().__init__()
        self.tile_sample_min_size = 384
        self.tile_sample_min_size_t = 33
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(hidden_size_mult) - 1)))
        
        # t_down_ratio = [i for i in encoder_temporal_downsample if len(i) > 0]
        # self.tile_latent_min_size_t = int((self.tile_sample_min_size_t-1) / (2 ** len(t_down_ratio))) + 1
        self.tile_latent_min_size_t = 9
        self.tile_overlap_factor = 0.125
        self.use_tiling = False

        self.learning_rate = lr
        self.lr_g_factor = 1.0

        self.loss = resolve_str_to_obj(loss_type, append=False)(
            **loss_params
        )

        self.encoder = Encoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=encoder_conv_in,
            conv_out=encoder_conv_out,
            attention=encoder_attention,
            resnet_blocks=encoder_resnet_blocks,
            spatial_downsample=encoder_spatial_downsample,
            temporal_downsample=encoder_temporal_downsample,
            mid_resnet=encoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            double_z=double_z,
        )

        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
        )

        quant_conv_cls = resolve_str_to_obj(q_conv)
        self.quant_conv = quant_conv_cls(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = quant_conv_cls(embed_dim, z_channels, 1)
        if hasattr(self.loss, "discriminator"):
            self.automatic_optimization = False

    def encode(self, x):
        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size
            or x.shape[-2] > self.tile_sample_min_size
            or x.shape[-3] > self.tile_sample_min_size_t
        ):
            return self.tiled_encode(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
            or z.shape[-3] > self.tile_latent_min_size_t
        ):
            return self.tiled_decode(z)
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

    def training_step(self, batch, batch_idx):
        if hasattr(self.loss, "discriminator"):
            return self._training_step_gan(batch, batch_idx=batch_idx)
        else:
            return self._training_step(batch, batch_idx=batch_idx)

    def _training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, "video")
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            split="train",
        )
        self.log(
            "aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )
        return aeloss

    def _training_step_gan(self, batch, batch_idx):
        inputs = self.get_input(batch, "video")
        reconstructions, posterior = self(inputs)
        opt1, opt2 = self.optimizers()
        
        # ---- AE Loss ----
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        opt1.zero_grad()
        self.manual_backward(aeloss)
        self.clip_gradients(opt1, gradient_clip_val=1, gradient_clip_algorithm="norm")
        opt1.step()
        # ---- GAN Loss ----
        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "discloss",
            discloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        opt2.zero_grad()
        self.manual_backward(discloss)
        self.clip_gradients(opt2, gradient_clip_val=1, gradient_clip_algorithm="norm")
        opt2.step()
        self.log_dict(
            {**log_dict_ae, **log_dict_disc},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

    def configure_optimizers(self):
        from itertools import chain

        lr = self.learning_rate
        modules_to_train = [
            self.encoder.named_parameters(),
            self.decoder.named_parameters(),
            self.post_quant_conv.named_parameters(),
            self.quant_conv.named_parameters(),
        ]
        params_with_time = []
        params_without_time = []
        for name, param in chain(*modules_to_train):
            if "time" in name:
                params_with_time.append(param)
            else:
                params_without_time.append(param)
        optimizers = []
        opt_ae = torch.optim.Adam(
            [
                {"params": params_with_time, "lr": lr},
                {"params": params_without_time, "lr": lr},
            ],
            lr=lr,
            betas=(0.5, 0.9),
        )
        optimizers.append(opt_ae)

        if hasattr(self.loss, "discriminator"):
            opt_disc = torch.optim.Adam(
                self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
            )
            optimizers.append(opt_disc)

        return optimizers, []

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_sample_min_size_t-1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i+1]+1] for i in range(len(t_chunk_idx)-1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        moments = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start: end]
            if idx != 0:
                moment = self.tiled_encode2d(chunk_x, return_moments=True)[:, :, 1:]
            else:
                moment = self.tiled_encode2d(chunk_x, return_moments=True)
            moments.append(moment)
        moments = torch.cat(moments, dim=2)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def tiled_decode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_latent_min_size_t-1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i+1]+1] for i in range(len(t_chunk_idx)-1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        dec_ = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start: end]
            if idx != 0:
                dec = self.tiled_decode2d(chunk_x)[:, :, 1:]
            else:
                dec = self.tiled_decode2d(chunk_x)
            dec_.append(dec)
        dec_ = torch.cat(dec_, dim=2)
        return dec_

    def tiled_encode2d(self, x, return_moments=False):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
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
        if return_moments:
            return moments
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
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
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

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def init_from_ckpt(self, path, ignore_keys=list(), remove_loss=False):
        sd = torch.load(path, map_location="cpu")
        print("init from " + path)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        
    def validation_step(self, batch, batch_idx):
        
        from ..utils.video_utils import tensor_to_video
        inputs = self.get_input(batch, 'video')
        latents = self.encode(inputs).sample()
        video_recon = self.decode(latents)
        for idx in range(len(video_recon)):
            self.logger.log_video(f"recon {batch_idx} {idx}", [tensor_to_video(video_recon[idx])], fps=[10])
            
        