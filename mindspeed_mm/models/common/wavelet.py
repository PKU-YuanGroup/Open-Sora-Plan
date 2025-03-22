import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from mindspeed_mm.models.common.conv import WfCausalConv3d
from mindspeed_mm.utils.utils import video_to_image


class HaarWaveletTransform3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.h_conv = WfCausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.g_conv = WfCausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.hh_conv = WfCausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.gh_conv = WfCausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.h_v_conv = WfCausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.g_v_conv = WfCausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.hh_v_conv = WfCausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.gh_v_conv = WfCausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        h = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) * 0.3536
        g = torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]) * 0.3536
        hh = torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]) * 0.3536
        gh = torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]) * 0.3536
        h_v = torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]) * 0.3536
        g_v = torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]) * 0.3536
        hh_v = torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]) * 0.3536
        gh_v = torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]) * 0.3536
        h = h.view(1, 1, 2, 2, 2)
        g = g.view(1, 1, 2, 2, 2)
        hh = hh.view(1, 1, 2, 2, 2)
        gh = gh.view(1, 1, 2, 2, 2)
        h_v = h_v.view(1, 1, 2, 2, 2)
        g_v = g_v.view(1, 1, 2, 2, 2)
        hh_v = hh_v.view(1, 1, 2, 2, 2)
        gh_v = gh_v.view(1, 1, 2, 2, 2)
        with torch.no_grad():
            self.h_conv.conv.weight.copy_(h.to(self.h_conv.conv.weight.device).to(self.h_conv.conv.weight.dtype))
            self.g_conv.conv.weight.copy_(g.to(self.g_conv.conv.weight.device).to(self.g_conv.conv.weight.dtype))
            self.hh_conv.conv.weight.copy_(hh.to(self.hh_conv.conv.weight.device).to(self.hh_conv.conv.weight.dtype))
            self.gh_conv.conv.weight.copy_(gh.to(self.gh_conv.conv.weight.device).to(self.gh_conv.conv.weight.dtype))
            self.h_v_conv.conv.weight.copy_(h_v.to(self.h_v_conv.conv.weight.device).to(self.h_v_conv.conv.weight.dtype))
            self.g_v_conv.conv.weight.copy_(g_v.to(self.g_v_conv.conv.weight.device).to(self.g_v_conv.conv.weight.dtype))
            self.hh_v_conv.conv.weight.copy_(hh_v.to(self.hh_v_conv.conv.weight.device).to(self.hh_v_conv.conv.weight.dtype))
            self.gh_v_conv.conv.weight.copy_(gh_v.to(self.gh_v_conv.conv.weight.device).to(self.gh_v_conv.conv.weight.dtype))
        
        self.h_conv.requires_grad_(False)
        self.g_conv.requires_grad_(False)
        self.hh_conv.requires_grad_(False)
        self.gh_conv.requires_grad_(False)
        self.h_v_conv.requires_grad_(False)
        self.g_v_conv.requires_grad_(False)
        self.hh_v_conv.requires_grad_(False)
        self.gh_v_conv.requires_grad_(False)

    def forward(self, x):
        assert x.dim() == 5
        b = x.shape[0]
        c = x.shape[1]
        
        x = rearrange(x, "b c t h w -> (b c) 1 t h w") # 3 1 17 256 256
        n_dim = x.shape[0]
        outputs = []
        for i in range(n_dim):
            y = x[i: i+1]
            outputs.append(self.h_conv(y))
            outputs.append(self.g_conv(y))
            outputs.append(self.hh_conv(y))
            outputs.append(self.gh_conv(y))
            outputs.append(self.h_v_conv(y))
            outputs.append(self.g_v_conv(y))
            outputs.append(self.hh_v_conv(y))
            outputs.append(self.gh_v_conv(y))
        
        outputs = torch.cat(outputs, dim=0)
        outputs = rearrange(outputs, "(b k c) 1 t h w -> b (c k) t h w", b=b, k=c)
        return outputs


class InverseHaarWaveletTransform3D(nn.Module):
    def __init__(self, enable_cached=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 'h' 'g' 'hh' 'gh' 'h_v' 'g_v' 'hh_v' 'gh_v'都是在小波变换中表示不同的滤波器filter的卷积核, h低通滤波, g高通滤波
        self.register_buffer('h',
            torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('g',
            torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('hh',
            torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('gh',
            torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('h_v',
            torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('g_v',
            torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('hh_v',
            torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('gh_v',
            torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.enable_cached = enable_cached
        self.is_first_chunk = True

    def forward(self, coeffs):
        if coeffs.dim() != 5:
            raise ValueError("coeffs.dim() must be 5.")
        b = coeffs.shape[0]

        (
            low_low_low,
            low_low_high,
            low_high_low,
            low_high_high,
            high_low_low,
            high_low_high,
            high_high_low,
            high_high_high,
        ) = coeffs.chunk(8, dim=1)

        low_low_low = rearrange(low_low_low, "b c t h w -> (b c) 1 t h w")
        low_low_high = rearrange(low_low_high, "b c t h w -> (b c) 1 t h w")
        low_high_low = rearrange(low_high_low, "b c t h w -> (b c) 1 t h w")
        low_high_high = rearrange(low_high_high, "b c t h w -> (b c) 1 t h w")
        high_low_low = rearrange(high_low_low, "b c t h w -> (b c) 1 t h w")
        high_low_high = rearrange(high_low_high, "b c t h w -> (b c) 1 t h w")
        high_high_low = rearrange(high_high_low, "b c t h w -> (b c) 1 t h w")
        high_high_high = rearrange(high_high_high, "b c t h w -> (b c) 1 t h w")

        low_low_low = F.conv_transpose3d(low_low_low, self.h, stride=2)
        low_low_high = F.conv_transpose3d(low_low_high, self.g, stride=2)
        low_high_low = F.conv_transpose3d(low_high_low, self.hh, stride=2)
        low_high_high = F.conv_transpose3d(low_high_high, self.gh, stride=2)
        high_low_low = F.conv_transpose3d(high_low_low, self.h_v, stride=2)
        high_low_high = F.conv_transpose3d(high_low_high, self.g_v, stride=2)
        high_high_low = F.conv_transpose3d(high_high_low, self.hh_v, stride=2)
        high_high_high = F.conv_transpose3d(high_high_high, self.gh_v, stride=2)
        
        if self.enable_cached and not self.is_first_chunk:
            reconstructed = (
                low_low_low
                + low_low_high
                + low_high_low
                + low_high_high
                + high_low_low
                + high_low_high
                + high_high_low
                + high_high_high
            )
        else:
            reconstructed = (
                low_low_low[:, :, 1:]
                + low_low_high[:, :, 1:]
                + low_high_low[:, :, 1:]
                + low_high_high[:, :, 1:]
                + high_low_low[:, :, 1:]
                + high_low_high[:, :, 1:]
                + high_high_low[:, :, 1:]
                + high_high_high[:, :, 1:]
            )
            
        reconstructed = rearrange(reconstructed, "(b c) 1 t h w -> b c t h w", b=b)
        return reconstructed


class HaarWaveletTransform2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('aa', torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('ad', torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('da', torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('dd', torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2)

    @video_to_image
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        low_low = F.conv2d(x, self.aa, stride=2).reshape(b, c, h // 2, w // 2)
        low_high = F.conv2d(x, self.ad, stride=2).reshape(b, c, h // 2, w // 2)
        high_low = F.conv2d(x, self.da, stride=2).reshape(b, c, h // 2, w // 2)
        high_high = F.conv2d(x, self.dd, stride=2).reshape(b, c, h // 2, w // 2)
        coeffs = torch.cat([low_low, low_high, high_low, high_high], dim=1)
        return coeffs


class InverseHaarWaveletTransform2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('aa', torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('ad', torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('da', torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('dd', torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2)

    @video_to_image
    def forward(self, coeffs):
        low_low, low_high, high_low, high_high = coeffs.chunk(4, dim=1)
        b, c, height_half, width_half = low_low.shape
        height = height_half * 2
        width = width_half * 2

        low_low = F.conv_transpose2d(
            low_low.reshape(b * c, 1, height_half, width_half), self.aa, stride=2
        )
        low_high = F.conv_transpose2d(
            low_high.reshape(b * c, 1, height_half, width_half), self.ad, stride=2
        )
        high_low = F.conv_transpose2d(
            high_low.reshape(b * c, 1, height_half, width_half), self.da, stride=2
        )
        high_high = F.conv_transpose2d(
            high_high.reshape(b * c, 1, height_half, width_half), self.dd, stride=2
        )

        return (low_low + low_high + high_low + high_high).reshape(b, c, height, width)