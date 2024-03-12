from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from timm.models.vision_transformer import Mlp
from torch.distributed import ProcessGroup

from opensora.models.diffusion.mmdit.common.modules.joint_attn import JointAttention

def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def modulate(norm_func, x, shift, scale, use_kernel=False):
    # Suppose x is (N, T, D), shift is (N, D), scale is (N, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    if use_kernel:
        try:
            from opendit.kernels.fused_modulate import fused_modulate

            x = fused_modulate(x, scale, shift)
        except ImportError:
            raise RuntimeError("FusedModulate kernel not available. Please install triton.")
    else:
        x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)

    return x

#################################################################################
#                                 Core MMDiT Model                                #
#################################################################################

class MMDiTBlock(nn.Module):
    """
    A MMDiT tansformer block for processing both latent and text embeddings.
    """
    def __init__(self, y_size, text_hidden_size, pix_hidden_size, num_heads, use_kernel=True, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        self.num_params = 6 # alpha-zeta
        self.text_y_input = nn.Sequential(
            nn.SiLU(),
            nn.Linear(y_size, self.num_params, bias=False)
        )
        self.pix_y_input = nn.Sequential(
            nn.SiLU(),
            nn.Linear(y_size, self.num_params, bias=False)
        )

        self.use_kernel = use_kernel
        self.text_norm1 = get_layernorm(text_hidden_size, 1e-6, False, self.use_kernel)
        self.pix_norm1 = get_layernorm(pix_hidden_size, 1e-6, False, self.use_kernel)

        self.attn = JointAttention(text_hidden_size, pix_hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.text_norm2 = get_layernorm(text_hidden_size, 1e-6, False, use_kernel)
        self.pix_norm2 = get_layernorm(pix_hidden_size, 1e-6, False, use_kernel)

        text_mlp_hidden_dim = int(text_hidden_size * mlp_ratio)
        pix_mlp_hidden_dim = int(pix_hidden_size * mlp_ratio)

        approx_gelu = lambda: nn.Identity() # act funcs are in the other places in SD3 to improve speed
        self.text_mlp = Mlp(in_features=text_hidden_size, hidden_features=text_mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.pix_mlp = Mlp(in_features=pix_hidden_size, hidden_features=pix_mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, y, x, c):

        def bc(tnsr):
            return tnsr.view(-1, 1, 1, 1, 1) # broadcasting for batched scalar-tensor multiplication

        coefs_text = self.text_y_input(y)
        coefs_pix = self.pix_y_input(y)

        # We use one linear layer pass and then splitting it into params for faster parallel inference
        alpha_txt, beta_txt, gamma_txt, delta_txt, eps_txt, dz_txt = coefs_text.unbind(-1)
        alpha_pix, beta_pix, gamma_pix, delta_pix, eps_pix, dz_pix = coefs_pix.unbind(-1)

        c = self.text_norm1(c)
        x = self.pix_norm1(x)

        c = bc(alpha_txt) * c + beta_txt * torch.ones_like(alpha_txt)
        x = bc(alpha_pix) * x + beta_pix * torch.ones_like(alpha_pix)

        x, c = self.attn(x, c)
        x_new = self.pix_norm2(x * bc(gamma_pix) + x) * bc(delta_pix) + eps_pix * torch.ones_like(x)
        c_new = self.text_norm2(c * bc(gamma_txt) + c) * bc(delta_txt) + eps_txt * torch.ones_like(c)

        x_new = self.pix_mlp(x_new) * bc(dz_pix) + x
        c_new = self.text_mlp(c_new) * bc(dz_txt) + c

        return x, c


class MMDiTFinalLayer(nn.Module):
    """
    The final layer of MMDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # TODO: adapt modulate's function
        x = modulate(nn.Identity(), x, shift, scale, use_kernel=self.use_kernel)
        x = self.linear(x)
        return x # NOTE: the processed text embedding is discarded here
