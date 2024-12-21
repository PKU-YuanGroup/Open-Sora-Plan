from typing import List, Literal
import torch
import torch.nn as nn
import os
from typing import Optional

from ..modules import (
    Downsample,
    ResnetBlock3D,
    Conv2d,
    HaarWaveletTransform3D,
    HaarWaveletTransform2D,
    InverseHaarWaveletTransform3D,
    InverseHaarWaveletTransform2D,
    CausalConv3d,
    Normalize,
    nonlinearity,
    Spatial2xTime2x3DDownsample,
    Spatial2xTime2x3DUpsample,
    Upsample,
    ResnetBlock2D
)
from ..registry import ModelRegistry
from ..modeling_videobase import VideoBaseAE
from ..utils.module_utils import resolve_str_to_obj
from ..utils.distrib_utils import DiagonalGaussianDistribution
from diffusers.configuration_utils import register_to_config

class WFDownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        down_type: Literal["thw", "hw"] = "thw",
        res_block: nn.Module = ResnetBlock3D,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size

        if down_type == "thw":
            self.wavelet_transform = HaarWaveletTransform3D()
            self.in_flow_conv = Conv2d(
                24, energy_flow_size, kernel_size=3, stride=1, padding=1
            )
            self.res_block = nn.Sequential(
                *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type) for _ in range(num_res_blocks - 1)]
            )
            self.down = Spatial2xTime2x3DDownsample(in_channels=in_channels, out_channels=in_channels)
            self.out_res_block = res_block(in_channels=in_channels + energy_flow_size, out_channels=out_channels, dropout=dropout, norm_type=norm_type)
            
        elif down_type == "hw":
            self.wavelet_transform = HaarWaveletTransform2D()
            self.in_flow_conv = Conv2d(
                12, energy_flow_size, kernel_size=3, stride=1, padding=1
            )
            self.res_block = nn.Sequential(
                *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type) for _ in range(num_res_blocks - 1)]
            )
            self.down = Downsample(in_channels=in_channels, out_channels=in_channels)
            self.out_res_block = res_block(in_channels=in_channels + energy_flow_size, out_channels=out_channels, dropout=dropout, norm_type=norm_type)

    def forward(self, x, w):
        x = self.res_block(x)
        x = self.down(x)
        
        coeffs = self.wavelet_transform(w[:, :3])
        w = self.in_flow_conv(coeffs)
        
        x = torch.concat([x, w], dim=1)
        return self.out_res_block(x), coeffs
        
class WFUpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        up_type: Literal["thw", "hw"] = "thw",
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        t_interpolation: str = "trilinear",
        res_block: nn.Module = ResnetBlock3D
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size
        assert num_res_blocks >= 2, "num res block too small"
        
        if up_type == "thw":
            self.branch_conv = ResnetBlock3D(in_channels=in_channels, out_channels=in_channels + energy_flow_size, dropout=dropout, norm_type=norm_type)
            self.out_flow_conv = nn.Sequential(
                ResnetBlock3D(in_channels=energy_flow_size, out_channels=energy_flow_size, dropout=dropout, norm_type=norm_type),
                Conv2d(in_channels=energy_flow_size, out_channels=24, kernel_size=3, stride=1, padding=1)
            )
            self.inverse_wavelet_transform = InverseHaarWaveletTransform3D()
            self.res_block = nn.Sequential(
                *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type) for _ in range(num_res_blocks - 2)]
            )
            self.up = Spatial2xTime2x3DUpsample(in_channels=in_channels, out_channels=in_channels, t_interpolation=t_interpolation)
            self.out_res_block = res_block(in_channels=in_channels, out_channels=out_channels, dropout=dropout, norm_type=norm_type)
            
        elif up_type == "hw":
            self.branch_conv = ResnetBlock2D(in_channels=in_channels, out_channels=in_channels + energy_flow_size, dropout=dropout, norm_type=norm_type)
            self.out_flow_conv = nn.Sequential(
                ResnetBlock2D(in_channels=energy_flow_size, out_channels=energy_flow_size, dropout=dropout, norm_type=norm_type),
                Conv2d(in_channels=energy_flow_size, out_channels=12, kernel_size=3, stride=1, padding=1)
            )
            self.inverse_wavelet_transform = InverseHaarWaveletTransform2D()
            self.res_block = nn.Sequential(
                *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type) for _ in range(num_res_blocks - 2)]
            )
            self.up = Upsample(in_channels=in_channels, out_channels=in_channels)
            self.out_res_block = res_block(in_channels=in_channels, out_channels=out_channels, dropout=dropout, norm_type=norm_type)
            
    def forward(self, x, w=None):
        x = self.branch_conv(x)
        
        coeffs = self.out_flow_conv(x[:, -self.energy_flow_size:])
        if w is not None:
            coeffs[:, :3] = coeffs[:, :3] + w
        w = self.inverse_wavelet_transform(coeffs)
        
        x = self.res_block(x[:, :-self.energy_flow_size])
        x = self.up(x)
        
        return self.out_res_block(x), w, coeffs

class Encoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        num_resblocks: int = 2,
        energy_flow_size: int = 64,
        dropout: float = 0.0,
        input_type: Literal["video", "image"] = "video",
        norm_type: str = "layernorm",
        base_channels: List[int] = [128, 256, 512],
        mid_layers_type: List[str] = ["ResnetBlock3D", "Attention3DFix","ResnetBlock3D"],
        down_layer_type: List[str] = ["hw", "thw"],
        down_layer_res_type: List[str] = ["ResnetBlock2D", "ResnetBlock3D"]
    ) -> None:
        super().__init__()
        self.input_type = input_type
        self.wavelet_transform_in = HaarWaveletTransform3D() if input_type == "video" else HaarWaveletTransform2D()
        self.conv_in = Conv2d(24 if input_type == "video" else 12, base_channels[0], kernel_size=3, stride=1, padding=1)
        
        self.down_blocks = nn.ModuleList()
        for idx, (down_type, down_res_type) in enumerate(zip(down_layer_type, down_layer_res_type)):
            down_block = WFDownBlock(
                in_channels=base_channels[idx],
                out_channels=base_channels[idx+1],
                energy_flow_size=energy_flow_size,
                num_res_blocks=num_resblocks,
                down_type=down_type,
                res_block=resolve_str_to_obj(down_res_type),
                dropout=dropout,
                norm_type=norm_type
            )
            self.down_blocks.append(down_block)
        
        # Mid
        mid_layers = []
        for mid_layer_type in mid_layers_type:
            if "Attn" in mid_layer_type:
                mid_layers.append(resolve_str_to_obj(mid_layer_type)(
                    in_channels=base_channels[-1], norm_type=norm_type,
                ))
            else:
                mid_layers.append(resolve_str_to_obj(mid_layer_type)(
                    in_channels=base_channels[-1], out_channels=base_channels[-1], dropout=dropout, norm_type=norm_type,
                ))
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = Normalize(base_channels[-1], norm_type=norm_type)
        if self.input_type == "video":
            self.conv_out = CausalConv3d(
                base_channels[-1], latent_dim * 2, kernel_size=3, stride=1, padding=1
            )
        else:
            self.conv_out = Conv2d(
                base_channels[-1], latent_dim * 2, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        coeffs = self.wavelet_transform_in(x)
        h = self.conv_in(coeffs)
        
        inter_coeffs = []
        for down_block in self.down_blocks:
            h, coeffs = down_block(h, coeffs)
            inter_coeffs.append(coeffs)
        
        h = self.mid(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h, inter_coeffs


class Decoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        num_resblocks: int = 2,
        dropout: float = 0.0,
        energy_flow_size: int = 128,
        input_type: Literal["video", "image"] = "video",
        norm_type: str = "layernorm",
        t_interpolation: str = "trilinear",
        base_channels: List[int] = [128, 256, 512],
        up_layer_type: List[str] = ["hw", "thw"],
        mid_layers_type: List[str] = ["ResnetBlock3D", "Attention3DFix","ResnetBlock3D"],
        up_layer_res_type: List[str] = ["ResnetBlock2D", "ResnetBlock3D"],
    ) -> None:
        super().__init__()
        self.energy_flow_size = energy_flow_size
        self.input_type = input_type
        self.conv_in = CausalConv3d(
            latent_dim, base_channels[-1], kernel_size=3, stride=1, padding=1
        ) if self.input_type == "video" else Conv2d(
            latent_dim, base_channels[-1], kernel_size=3, stride=1, padding=1
        )
        
        mid_layers = []
        for mid_layer_type in mid_layers_type:
            if "Attn" in mid_layer_type:
                mid_layers.append(resolve_str_to_obj(mid_layer_type)(
                    in_channels=base_channels[-1],
                    norm_type=norm_type,
                ))
            else:
                mid_layers.append(resolve_str_to_obj(mid_layer_type)(
                    in_channels=base_channels[-1],
                    out_channels=base_channels[-1],
                    dropout=dropout,
                    norm_type=norm_type,
                ))
        self.mid = nn.Sequential(*mid_layers)

        self.up_blocks = nn.ModuleList()
        for idx, (up_type, up_res_type) in enumerate(zip(reversed(up_layer_type), reversed(up_layer_res_type))):
            idx = len(up_layer_type) - idx
            up_block = WFUpBlock(
                in_channels=base_channels[idx],
                out_channels=base_channels[idx-1],
                energy_flow_size=energy_flow_size,
                num_res_blocks=num_resblocks,
                up_type=up_type,
                res_block=resolve_str_to_obj(up_res_type),
                t_interpolation=t_interpolation,
                dropout=dropout,
                norm_type=norm_type
            )
            self.up_blocks.append(up_block)
        
        # Out
        self.norm_out = Normalize(base_channels[0], norm_type=norm_type)
        self.conv_out = Conv2d(base_channels[0], 24 if input_type == "video" else 12, kernel_size=3, stride=1, padding=1)
        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform3D() if self.input_type == "video" else InverseHaarWaveletTransform2D()

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        inter_coeffs = []
        w = None
        for up_block in self.up_blocks:
            h, w, coeffs = up_block(h, w)
            inter_coeffs.append(coeffs)
        h = self.norm_out(h)
        h = self.conv_out(h)
        h[:, :3]= h[:, :3] + w
        dec = self.inverse_wavelet_transform_out(h)
        return dec, inter_coeffs


@ModelRegistry.register("WFVAE2")
class WFVAE2Model(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 4,
        base_channels: List[int] = [128, 256, 512],
        decoder_base_channels: Optional[List[int]] = None,
        layer_type: List[str] = ["hw", "thw"],
        layer_res_type: List[str] = ["ResnetBlock2D", "ResnetBlock3D"],
        encoder_num_resblocks: int = 2,
        encoder_energy_flow_size: int = 128,
        decoder_num_resblocks: int = 3,
        decoder_energy_flow_size: int = 128,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        t_interpolation: str = "trilinear",
        input_type: Literal["video", "image"] = "video",
        mid_layers_type: List[str] = ["ResnetBlock3D" ,"ResnetBlock3D"],
        scale: List[float] = [0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215],
        shift: List[float] = [0, 0, 0, 0, 0, 0, 0, 0],
    ) -> None:
        super().__init__()
        self.use_tiling = False
        # Hardcode for now
        self.t_chunk_enc = 16
        self.t_chunk_dec = 4
        self.use_quant_layer = False

        if decoder_base_channels is None:
            decoder_base_channels = base_channels
        
        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=encoder_num_resblocks,
            energy_flow_size=encoder_energy_flow_size,
            dropout=dropout,
            norm_type=norm_type,
            down_layer_type=layer_type,
            down_layer_res_type=layer_res_type,
            input_type=input_type,
            mid_layers_type=mid_layers_type
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=decoder_base_channels,
            num_resblocks=decoder_num_resblocks,
            energy_flow_size=decoder_energy_flow_size,
            dropout=dropout,
            norm_type=norm_type,
            t_interpolation=t_interpolation,
            up_layer_type=layer_type,
            up_layer_res_type=layer_res_type,
            input_type=input_type,
            mid_layers_type=mid_layers_type
        )

    def get_encoder(self):
        return [self.encoder]

    def get_decoder(self):
        return [self.decoder]

    def encode(self, x):
        h, coeffs = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        dec, coeffs = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        encode_output = self.encode(input)
        posterior, enc_coeffs = (
            encode_output.latent_dist,
            encode_output.extra_output,
        )

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        decode_output = self.decode(z)
        dec, dec_coeffs = decode_output.sample, decode_output.extra_output
        
        return dec, posterior

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

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