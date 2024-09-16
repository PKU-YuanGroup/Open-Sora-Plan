try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
    
from ..modeling_videobase import VideoBaseAE
from diffusers.configuration_utils import register_to_config
import torch
import torch.nn as nn
from ..modules import (
    ResnetBlock2D,
    ResnetBlock3D,
    Conv2d,
    Downsample,
    Upsample,
    Spatial2xTime2x3DDownsample,
    Spatial2xTime2x3DUpsample,
    CausalConv3d,
    Normalize,
    AttnBlock3DFix,
    nonlinearity,
)
import torch.nn as nn
from ..utils.distrib_utils import DiagonalGaussianDistribution
from ..utils.wavelet_utils import (
    HaarWaveletTransform2D,
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform2D,
    InverseHaarWaveletTransform3D
)
import torch
from copy import deepcopy
import os
from ..registry import ModelRegistry
from einops import rearrange


class Encoder(VideoBaseAE):

    @register_to_config
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
        super().__init__()
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
                )
                for _ in range(num_resblocks)
            ],
            Spatial2xTime2x3DDownsample(base_channels * 2, base_channels * 2),
        )
        # Connection
        self.connect_l2 = Conv2d(
            12, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        self.connect_l3 = Conv2d(
            24, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        # Mid
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 2 + energy_flow_hidden_size,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, AttnBlock3DFix(in_channels=base_channels * 4, norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = Normalize(base_channels * 4, norm_type=norm_type)
        self.conv_out = CausalConv3d(
            base_channels * 4, latent_dim * 2, kernel_size=3, stride=1, padding=1
        )
        
        self.wavelet_tranform_3d = HaarWaveletTransform3D()
        self.wavelet_tranform_2d = HaarWaveletTransform2D()
        
        
    def forward(self, coeffs):
        l2_coeffs = coeffs[:, :3]
        t = l2_coeffs.shape[2]
        l2_coeffs = rearrange(l2_coeffs, "b c t h w -> (b t) c h w")
        l2_coeffs = self.wavelet_tranform_2d(l2_coeffs)
        l2_coeffs = rearrange(l2_coeffs, "(b t) c h w -> b c t h w", t=t)
        l2 = self.connect_l2(l2_coeffs)
        l3_coeffs = self.wavelet_tranform_3d(l2_coeffs[:, :3])
        l3 = self.connect_l3(l3_coeffs)
        
        h = self.down1(coeffs)
        h = torch.concat([h, l2], dim=1)
        h = self.down2(h)
        h = torch.concat([h, l3], dim=1)
        h = self.mid(h)
        
        if npu_config is None:
            h = self.norm_out(h)
        else:
            h = npu_config.run_group_norm(self.norm_out, h)
            
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(VideoBaseAE):

    @register_to_config
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
        super().__init__()
        self.energy_flow_hidden_size = energy_flow_hidden_size
    
        self.conv_in = CausalConv3d(
            latent_dim, base_channels * 4, kernel_size=3, stride=1, padding=1
        )
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, AttnBlock3DFix(in_channels=base_channels * 4, norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)

        self.up2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            Spatial2xTime2x3DUpsample(
                base_channels * 4, base_channels * 4, t_interpolation=t_interpolation
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )
        self.up1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (4 if i == 0 else 2),
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(num_resblocks)
            ],
            Upsample(in_channels=base_channels * 2, out_channels=base_channels * 2),
            ResnetBlock3D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )
        self.layer = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (2 if i == 0 else 1),
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(2)
            ],
        )
        # Connection
        self.connect_l2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(base_channels, 12, kernel_size=3, stride=1, padding=1),
        )
        self.connect_l3 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1),
        )
        # Out
        self.norm_out = Normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1)

        self.inverse_wavelet_tranform_3d = InverseHaarWaveletTransform3D()
        self.inverse_wavelet_tranform_2d = InverseHaarWaveletTransform2D()
        
        
    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        l3_coeffs = self.connect_l3(h[:, -self.energy_flow_hidden_size :])
        l3 = self.inverse_wavelet_tranform_3d(l3_coeffs)
        h = self.up2(h[:, : -self.energy_flow_hidden_size])
        l2_coeffs = h[:, -self.energy_flow_hidden_size :]
        l2_coeffs = self.connect_l2(l2_coeffs)
        l2_coeffs[:, :3] = l2_coeffs[:, :3] + l3
        
        t = l2_coeffs.shape[2]
        l2_coeffs = rearrange(l2_coeffs, "b c t h w -> (b t) c h w")
        l2 = self.inverse_wavelet_tranform_2d(l2_coeffs)
        l2 = rearrange(l2, "(b t) c h w -> b c t h w", t=t)

        h = self.up1(h[:, : -self.energy_flow_hidden_size])
        
        h = self.layer(h)
        if npu_config is None:
            h = self.norm_out(h)
        else:
            h = npu_config.run_group_norm(self.norm_out, h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h[:, :3] = h[:, :3] + l2
        return h


@ModelRegistry.register("WFVAE")
class WFVAEModel(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        self.use_tiling = False
        # Hardcode for now
        self.t_chunk_enc = 16
        self.t_upsample_times = 4 // 2
        self.t_chunk_dec = 2
        self.use_quant_layer = False

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
        )

        # Set cache offset for trilinear lossless upsample.
        self._set_cache_offset([self.decoder.up2, self.decoder.connect_l3, self.decoder.conv_in, self.decoder.mid], 1)
        self._set_cache_offset([self.decoder.up2[-2:], self.decoder.up1, self.decoder.connect_l2, self.decoder.layer], self.t_upsample_times)
        
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
                module.causal_cached = None
                
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
            end = min(t, end + (self.t_chunk_dec if decoder_mode else self.t_chunk_enc) )
            start_end.append([start, end])
            start = end
        return start_end
    
    def encode(self, x):
        self._empty_causal_cached(self.encoder)
        
        if torch_npu is not None:
            dtype = x.dtype
            x = x.to(torch.float16)
            wt = HaarWaveletTransform3D().to(x.device, dtype=x.dtype)
            coeffs = wt(x)
            coeffs = coeffs.to(dtype)
        else:
            wt = HaarWaveletTransform3D().to(x.device, dtype=x.dtype)
            coeffs = wt(x)
            
        if self.use_tiling:
            h = self.tile_encode(coeffs)
        else:
            h = self.encoder(coeffs)
            if self.use_quant_layer:
                h = self.quant_conv(h)
            
        posterior = DiagonalGaussianDistribution(h)
        return posterior
    
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
        self._empty_causal_cached(self.decoder)
        
        if self.use_tiling:
            dec = self.tile_decode(z)
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec = self.decoder(z)
        if torch_npu is not None:
            dtype = dec.dtype
            dec = dec.to(torch.float16)
            wt = InverseHaarWaveletTransform3D().to(dec.device, dtype=dec.dtype)
            dec = wt(dec)
            dec = dec.to(dtype)
        else:
            wt = InverseHaarWaveletTransform3D().to(dec.device, dtype=dec.dtype)
            dec = wt(dec)
            
        return dec
    
    def tile_decode(self, x):
        b, c, t, h, w = x.shape
        
        start_end = self.build_chunk_start_end(t, decoder_mode=True)
        
        result = []
        for start, end in start_end:
            if end + 1 < t:
                chunk = x[:, :, start:end+1, :, :]
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

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
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

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling
        self._set_causal_cached(use_tiling)
        
    def disable_tiling(self):
        self.enable_tiling(False)

    def init_from_ckpt(self, path, ignore_keys=list()):
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