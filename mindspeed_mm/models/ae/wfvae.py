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


WFVAE_MODULE_MAPPINGS = {
    "Conv2d": Conv2d,
    "ResnetBlock2D": ResnetBlock2D,
    "CausalConv3d": WfCausalConv3d,
    "AttnBlock3D": WfCausalConv3dAttnBlock,
    "ResnetBlock3D": ResnetBlock3D,
    "Downsample": Downsample,
    "HaarWaveletTransform2D": HaarWaveletTransform2D,
    "HaarWaveletTransform3D": HaarWaveletTransform3D,
    "InverseHaarWaveletTransform2D": InverseHaarWaveletTransform2D,
    "InverseHaarWaveletTransform3D": InverseHaarWaveletTransform3D,
    "Spatial2xTime2x3DDownsample": Spatial2xTime2x3DDownsample,
    "Upsample": Upsample,
    "Spatial2xTime2x3DUpsample": CachedCausal3DUpsample
}

def model_name_to_cls(model_name):
    if model_name in WFVAE_MODULE_MAPPINGS:
        return WFVAE_MODULE_MAPPINGS[model_name]
    else:
        raise ValueError(f"Model name {model_name} not supported")

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
            l1_dowmsample_block: str = "Downsample",
            l1_downsample_wavelet: str = "HaarWaveletTransform2D",
            l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
            l2_downsample_wavelet: str = "HaarWaveletTransform3D",
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
            model_name_to_cls(l1_dowmsample_block)(in_channels=base_channels, out_channels=base_channels),
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
            model_name_to_cls(l2_dowmsample_block)(base_channels * 2, base_channels * 2, conv_type="WfCausalConv3d"),
        )
        # Connection
        if l1_downsample_wavelet.endswith("2D"):
            l1_channels = 12
        else:
            l1_channels = 24
        self.connect_l1 = Conv2d(
            l1_channels, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
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

        self.wavelet_transform_in = HaarWaveletTransform3D()
        self.wavelet_transform_l1 = model_name_to_cls(l1_downsample_wavelet)()
        self.wavelet_transform_l2 = model_name_to_cls(l2_downsample_wavelet)()

    def forward(self, x):
        coeffs = self.wavelet_transform_in(x)
        
        l1_coeffs = coeffs[:, :3]
        l1_coeffs = self.wavelet_transform_l1(l1_coeffs)
        l1 = self.connect_l1(l1_coeffs)
        l2_coeffs = self.wavelet_transform_l2(l1_coeffs[:, :3])
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
            connect_res_layer_num: int = 2,
            l1_upsample_block: str = "Upsample",
            l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
            l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
            l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
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
            model_name_to_cls(l2_upsample_block)(
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
            model_name_to_cls(l1_upsample_block)(in_channels=base_channels * 2, out_channels=base_channels * 2),
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
        if l1_upsample_wavelet.endswith("2D"):
            l1_channels = 12
        else:
            l1_channels = 24
        self.connect_l1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(energy_flow_hidden_size, l1_channels, kernel_size=3, stride=1, padding=1),
        )
        self.connect_l2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    conv_type="WfCausalConv3d"
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(energy_flow_hidden_size, 24, kernel_size=3, stride=1, padding=1),
        )
        # Out
        self.norm_out = normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1)

        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform3D()
        self.inverse_wavelet_transform_l1 = model_name_to_cls(l1_upsample_wavelet)()
        self.inverse_wavelet_transform_l2 = model_name_to_cls(l2_upsample_wavelet)()

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        l2_coeffs = self.connect_l2(h[:, -self.energy_flow_hidden_size :])
        l2 = self.inverse_wavelet_transform_l1(l2_coeffs)
        h = self.up2(h[:, : -self.energy_flow_hidden_size])

        l1_coeffs = h[:, -self.energy_flow_hidden_size :]
        l1_coeffs = self.connect_l1(l1_coeffs)
        l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
        l1 = self.inverse_wavelet_transform_l2(l1_coeffs)

        h = self.up1(h[:, : -self.energy_flow_hidden_size])
        h = self.layer(h)
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        h[:, :3] = h[:, :3] + l1
        dec = self.inverse_wavelet_transform_out(h)
        return dec


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
            l1_dowmsample_block: str = "Downsample",
            l1_downsample_wavelet: str = "HaarWaveletTransform2D",
            l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
            l2_downsample_wavelet: str = "HaarWaveletTransform3D",
            l1_upsample_block: str = "Upsample",
            l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
            l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
            l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
            **kwargs
    ) -> None:
        super().__init__(config=None)

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
            l1_dowmsample_block=l1_dowmsample_block,
            l1_downsample_wavelet=l1_downsample_wavelet,
            l2_dowmsample_block=l2_dowmsample_block,
            l2_downsample_wavelet=l2_downsample_wavelet
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
            connect_res_layer_num=connect_res_layer_num,
            l1_upsample_block=l1_upsample_block,
            l1_upsample_wavelet=l1_upsample_wavelet,
            l2_upsample_block=l2_upsample_block,
            l2_upsample_wavelet=l2_upsample_wavelet
        )

        # Set cache offset for trilinear lossless upsample.
        if l1_dowmsample_block == "Downsample":
            self.temporal_uptimes = 4
            self._set_cache_offset([self.decoder.up2, self.decoder.connect_l2, self.decoder.conv_in, self.decoder.mid], 1)
            self._set_cache_offset([self.decoder.up2[-2:], self.decoder.up1, self.decoder.connect_l1, self.decoder.layer], 2)
        else:
            self.temporal_uptimes = 8
            self._set_cache_offset([self.decoder.up2, self.decoder.connect_l2, self.decoder.conv_in, self.decoder.mid], 1)
            self._set_cache_offset([self.decoder.up2[-2:], self.decoder.connect_l1, self.decoder.up1], 2)
            self._set_cache_offset([self.decoder.up1[-2:], self.decoder.layer], 4)
        
        self.enable_tiling(use_tiling)

        if from_pretrained is not None:
            self.init_from_ckpt(from_pretrained)
        
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
        dtype = x.dtype
        self._empty_causal_cached(self.encoder)
        self._set_first_chunk(True)
        
        if self.use_tiling:
            h = self.tile_encode(x)
        else:
            h = self.encoder(x)
            if self.use_quant_layer:
                h = self.quant_conv(h)
            
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def tile_encode(self, x):
        b, c, t, h, w = x.shape
        
        start_end = self.build_chunk_start_end(t)
        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            chunk = x[:, :, start:end, :, :]
            chunk = self.encoder(chunk)
            if self.use_quant_layer:
                chunk = self.quant_conv(chunk)
            result.append(chunk)
            
        return torch.cat(result, dim=2)

    def decode(self, z):
        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)
        
        if self.use_tiling:
            dec = self.tile_decode(z)
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec = self.decoder(z) 
        return dec

    def tile_decode(self, x):
        b, c, t, h, w = x.shape
        
        start_end = self.build_chunk_start_end(t, decoder_mode=True)
        
        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx==0)
            
            if end + 1 < t:
                chunk = x[:, :, start:end+1, :, :]
            else:
                chunk = x[:, :, start:end, :, :]
                
            if self.use_quant_layer:
                chunk = self.post_quant_conv(chunk)
            chunk = self.decoder(chunk)
            
            if end + 1 < t:
                chunk = chunk[:, :, :-self.temporal_uptimes]
                result.append(chunk.clone())
            else:
                result.append(chunk.clone())
            
        return torch.cat(result, dim=2)

    def forward(self, x):
        z = self.encode(x).sample()
        dec = self.decode(z)
        return dec

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