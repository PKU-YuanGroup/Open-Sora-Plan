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
    haar_wavelet_transform_3d,
    inverse_haar_wavelet_transform_3d,
    haar_wavelet_transform_2d_new,
    inverse_haar_wavelet_transform_2d_new,
)
import torch
from copy import deepcopy
import os
from .modeling_causalvae import Decoder
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
            Conv2d(24, 128, kernel_size=3, stride=1, padding=1),
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
                in_channels=base_channels * 2 + 64,
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
            mid_layers.insert(1, AttnBlock3DFix(in_channels=base_channels * 4, norm_type=norm_type))
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = Normalize(base_channels * 4, norm_type=norm_type)
        self.conv_out = CausalConv3d(
            base_channels * 4, latent_dim * 2, kernel_size=3, stride=1, padding=1
        )

    def forward(self, coeffs):
        l2_coeffs = coeffs[:, :3]
        t = l2_coeffs.shape[2]
        l2_coeffs = rearrange(l2_coeffs, "b c t h w -> (b t) c h w")
        l2_coeffs = haar_wavelet_transform_2d_new(l2_coeffs)
        l2_coeffs = rearrange(l2_coeffs, "(b t) c h w -> b c t h w", t=t)
        l2 = self.connect_l2(l2_coeffs)
        l3_coeffs = haar_wavelet_transform_3d(l2_coeffs[:, :3])
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
            mid_layers.insert(1, AttnBlock3DFix(in_channels=base_channels * 4, norm_type=norm_type))
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
                for _ in range(2)
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
                for _ in range(2)
            ],
            Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1),
        )
        # Out
        self.norm_out = Normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)

        l3_coeffs = self.connect_l3(h[:, -self.energy_flow_hidden_size :])
        l3 = inverse_haar_wavelet_transform_3d(l3_coeffs)

        h = self.up2(h[:, : -self.energy_flow_hidden_size])

        l2_coeffs = h[:, -self.energy_flow_hidden_size :]
        l2_coeffs = self.connect_l2(l2_coeffs)
        l2_coeffs[:, :3] = l2_coeffs[:, :3] + l3
        t = l2_coeffs.shape[2]
        l2_coeffs = rearrange(l2_coeffs, "b c t h w -> (b t) c h w")
        l2 = inverse_haar_wavelet_transform_2d_new(l2_coeffs)
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

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def encode(self, x):
        coeffs = haar_wavelet_transform_3d(x)
        h = self.encoder(coeffs)
        if self.use_quant_layer:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def _is_wavelet_output(self, x):
        return x.shape[1] != 3

    def decode(self, z):
        if self.use_quant_layer:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if self._is_wavelet_output(dec):
            dec = inverse_haar_wavelet_transform_3d(dec)
        return dec

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
        raise NotImplementedError("WFVAE not support tiling yet.")
        self.use_tiling = use_tiling

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

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=True)
        print(missing_keys, unexpected_keys)
