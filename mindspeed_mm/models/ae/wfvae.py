import os
from collections import deque

import torch.nn as nn
import torch

from mindspeed_mm.models.common.checkpoint import load_checkpoint
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.attention import WfCausalConv3dAttnBlock
from mindspeed_mm.models.common.distrib import DiagonalGaussianDistribution
from mindspeed_mm.models.common.wavelet import (
    HaarWaveletTransform2D,
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform2D,
    InverseHaarWaveletTransform3D
)
from mindspeed_mm.models.common.conv import Conv2d, WfCausalConv3d
from mindspeed_mm.models.common.resnet_block import ResnetBlock2D, ResnetBlock3D
from mindspeed_mm.models.common.updownsample import Upsample, Downsample, Spatial2xTime2x3DDownsample, CachedCausal3DUpsample


class Encoder(MultiModalModule):

    def __init__(
            self,
            latent_dim: int = 8,
            base_channels: int = 128,
            num_resblocks: int = 2,
            energy_flow_hidden_size: int = 64,
            dropout: float = 0.0,
            use_attention: bool = True,
            norm_type: str = "groupnorm",
    ) -> None:
        super().__init__(config=None)
        self.activation = nn.SiLU()
        self.down1 = nn.Sequential(
            Conv2d(24, base_channels, kernel_size=3, stride=1, padding=1),
            *[
                ResnetBlock2D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            Downsample(in_channels=base_channels, out_channels=base_channels),
        )
        self.down2 = nn.Sequential(
            Conv2d(
                base_channels + energy_flow_hidden_size,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 2,
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for _ in range(num_resblocks)
            ],
            Spatial2xTime2x3DDownsample(base_channels * 2, base_channels * 2, conv_type="WfCausalConv3d"),
        )
        # Connection
        self.connect_l1 = Conv2d(
            12, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        self.connect_l2 = Conv2d(
            24, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        # Mid
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 2 + energy_flow_hidden_size,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, WfCausalConv3dAttnBlock(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = normalize(base_channels * 4, norm_type=norm_type)
        self.conv_out = WfCausalConv3d(
            base_channels * 4, latent_dim * 2, kernel_size=3, stride=1, padding=1
        )

        self.wavelet_tranform_l1 = HaarWaveletTransform2D()
        self.wavelet_tranform_l2 = HaarWaveletTransform3D()

    def forward(self, coeffs):
        l1_coeffs = coeffs[:, :3]
        l1_coeffs = self.wavelet_tranform_l1(l1_coeffs)
        l1 = self.connect_l1(l1_coeffs)
        l2_coeffs = self.wavelet_tranform_l2(l1_coeffs[:, :3])
        l2 = self.connect_l2(l2_coeffs)

        h = self.down1(coeffs)
        h = torch.concat([h, l1], dim=1)
        h = self.down2(h)
        h = torch.concat([h, l2], dim=1)
        h = self.mid(h)

        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h


class Decoder(MultiModalModule):

    def __init__(
            self,
            latent_dim: int = 8,
            base_channels: int = 128,
            num_resblocks: int = 2,
            dropout: float = 0.0,
            energy_flow_hidden_size: int = 128,
            use_attention: bool = True,
            norm_type: str = "groupnorm",
            t_interpolation: str = "nearest",
            connect_res_layer_num: int = 2
    ) -> None:
        super().__init__(config=None)
        self.energy_flow_hidden_size = energy_flow_hidden_size
        self.activation = nn.SiLU()
        self.conv_in = WfCausalConv3d(
            latent_dim, base_channels * 4, kernel_size=3, stride=1, padding=1
        )
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, WfCausalConv3dAttnBlock(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)

        self.up2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for _ in range(num_resblocks)
            ],
            CachedCausal3DUpsample(
                base_channels * 4, base_channels * 4, t_interpolation=t_interpolation
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
        )
        self.up1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (4 if i == 0 else 2),
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for i in range(num_resblocks)
            ],
            Upsample(in_channels=base_channels * 2, out_channels=base_channels * 2),
            ResnetBlock3D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                dropout=dropout,
                norm_type=norm_type,
                conv_type="WfCausalConv3d"
            ),
        )
        self.layer = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (2 if i == 0 else 1),
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for i in range(2)
            ],
        )
        # Connection
        self.connect_l1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(base_channels, 12, kernel_size=3, stride=1, padding=1),
        )
        self.connect_l2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1),
        )
        # Out
        self.norm_out = normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1)

        self.inverse_wavelet_tranform_l1 = InverseHaarWaveletTransform2D()
        self.inverse_wavelet_tranform_l2 = InverseHaarWaveletTransform3D()

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        l2_coeffs = self.connect_l2(h[:, -self.energy_flow_hidden_size :])
        l2 = self.inverse_wavelet_tranform_l2(l2_coeffs)
        h = self.up2(h[:, : -self.energy_flow_hidden_size])

        l1_coeffs = h[:, -self.energy_flow_hidden_size :]
        l1_coeffs = self.connect_l1(l1_coeffs)
        l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
        l1 = self.inverse_wavelet_tranform_l1(l1_coeffs)

        h = self.up1(h[:, : -self.energy_flow_hidden_size])
        h = self.layer(h)
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        h[:, :3] = h[:, :3] + l1
        return h


class WFVAE(MultiModalModule):

    def __init__(
            self,
            from_pretrained: str = None,
            latent_dim: int = 8,
            base_channels: int = 128,
            encoder_num_resblocks: int = 2,
            encoder_energy_flow_hidden_size: int = 64,
            decoder_num_resblocks: int = 2,
            decoder_energy_flow_hidden_size: int = 128,
            use_attention: bool = True,
            dropout: float = 0.0,
            norm_type: str = "groupnorm",
            t_interpolation: str = "nearest",
            vae_scale_factor: list = None,
            use_tiling: bool = False,
            connect_res_layer_num: int = 2,
            **kwargs
    ) -> None:
        super().__init__(config=None)

        self.register_buffer("scale", torch.tensor([0.18215] * 8)[None, :, None, None, None])
        self.register_buffer("shift", torch.zeros(1, 8, 1, 1, 1))

        # Hardcode for now
        self.t_chunk_enc = 16
        self.t_upsample_times = 4 // 2
        self.t_chunk_dec = 4
        self.use_quant_layer = False
        self.vae_scale_factor = vae_scale_factor

        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=encoder_num_resblocks,
            energy_flow_hidden_size=encoder_energy_flow_hidden_size,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=decoder_num_resblocks,
            energy_flow_hidden_size=decoder_energy_flow_hidden_size,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            t_interpolation=t_interpolation,
            connect_res_layer_num=connect_res_layer_num
        )

        # Set cache offset for trilinear lossless upsample.
        self._set_cache_offset([self.decoder.up2, self.decoder.connect_l2, self.decoder.conv_in, self.decoder.mid], 1)
        self._set_cache_offset([self.decoder.up2[-2:], self.decoder.up1, self.decoder.connect_l1, self.decoder.layer], self.t_upsample_times)

        if from_pretrained is not None:
            load_checkpoint(self, from_pretrained)

        self.enable_tiling(use_tiling)

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def _empty_causal_cached(self, parent):
        for name, module in parent.named_modules():
            if hasattr(module, 'causal_cached'):
                module.causal_cached = deque()

    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, 'is_first_chunk'):
                module.is_first_chunk = is_first_chunk

    def _set_causal_cached(self, enable_cached=True):
        for name, module in self.named_modules():
            if hasattr(module, 'enable_cached'):
                module.enable_cached = enable_cached

    def _set_cache_offset(self, modules, cache_offset=0):
        for module in modules:
            for submodule in module.modules():
                if hasattr(submodule, 'cache_offset'):
                    submodule.cache_offset = cache_offset

    def build_chunk_start_end(self, t, decoder_mode=False):
        start_end = [[0, 1]]
        start = 1
        end = start
        while True:
            if start >= t:
                break
            end = min(t, end + (self.t_chunk_dec if decoder_mode else self.t_chunk_enc))
            start_end.append([start, end])
            start = end
        return start_end

    def encode(self, x):
        self._empty_causal_cached(self.encoder)
        dtype = x.dtype
        wt = HaarWaveletTransform3D().to(x.device, dtype=x.dtype)
        coeffs = wt(x)

        if self.use_tiling:
            h = self.tile_encode(coeffs)
        else:
            h = self.encoder(coeffs)
            if self.use_quant_layer:
                h = self.quant_conv(h)

        posterior = DiagonalGaussianDistribution(h)
        return (posterior.sample() - self.shift.to(x.device, dtype=dtype)) * self.scale.to(x.device, dtype)

    def tile_encode(self, x):
        b, c, t, h, w = x.shape

        start_end = self.build_chunk_start_end(t)
        result = []
        for start, end in start_end:
            chunk = x[:, :, start:end, :, :]
            chunk = self.encoder(chunk)
            if self.use_quant_layer:
                chunk = self.encoder(chunk)
            result.append(chunk)

        return torch.cat(result, dim=2)

    def decode(self, z):
        z /= z / self.scale.to(z.device, dtype=z.dtype) + self.shift.to(z.device, dtype=z.dtype)
        self._empty_causal_cached(self.decoder)

        if self.use_tiling:
            dec = self.tile_decode(z)
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec = self.decoder(z)

        dtype = dec.dtype
        dec = dec.to(torch.float16)
        wt = InverseHaarWaveletTransform3D().to(dec.device, dtype=dec.dtype)
        dec = wt(dec)
        dec = dec.to(dtype)

        return dec

    def tile_decode(self, x):
        b, c, t, h, w = x.shape

        start_end = self.build_chunk_start_end(t, decoder_mode=True)

        result = []
        for start, end in start_end:
            if end + 1 < t:
                chunk = x[:, :, start:end + 1, :, :]
            else:
                chunk = x[:, :, start:end, :, :]

            if self.use_quant_layer:
                chunk = self.post_quant_conv(chunk)
            chunk = self.decoder(chunk)

            if end + 1 < t:
                chunk = chunk[:, :, :-2]
                result.append(chunk.clone())
            else:
                result.append(chunk.clone())

        return torch.cat(result, dim=2)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def enable_tiling(self, use_tiling: bool = False):
        self.use_tiling = use_tiling
        self._set_causal_cached(use_tiling)

    def disable_tiling(self):
        self.enable_tiling(False)

    def init_from_ckpt(self, path, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = list()
        sd = torch.load(path, map_location="cpu")
        print("init from " + path)

        if (
                "ema_state_dict" in sd
                and len(sd["ema_state_dict"]) > 0
                and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0
        ):
            print("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            print("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]

        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(missing_keys, unexpected_keys)