import torch.nn as nn
import torch.nn.functional as F
from .normalize import Normalize
from .conv import CausalConv3d
import torch
from .block import Block

try:
    import torch_npu
    from opensora.npu_config import npu_config, set_run_dtype
except:
    torch_npu = None
    npu_config = None
    # from xformers import ops as xops

class AttnBlock3D(Block):
    """Compatible with old versions, there are issues, use with caution."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = q.reshape(b * t, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * t, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b * t, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, t, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class AttnBlock3DFix(nn.Module):
    """
    Thanks to https://github.com/PKU-YuanGroup/Open-Sora-Plan/pull/172.
    """
    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        
        if torch_npu is None:
            # attn_output = xops.memory_efficient_attention(
            #     q, k, v,
            #     scale=c ** -0.5
            # )
            q = q.view(b * t, -1, 1, c).transpose(1, 2)
            k = k.view(b * t, -1, 1, c).transpose(1, 2)
            v = v.view(b * t, -1, 1, c).transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).reshape(b * t, -1, 1 * c)

        else:
            # print('npu_config.enable_FA, q.dtype == torch.float32', npu_config.enable_FA, q.dtype == torch.float32)
            if npu_config.enable_FA and q.dtype == torch.float32:
                dtype = torch.bfloat16
            else:
                dtype = None
            with set_run_dtype(q, dtype):
                query, key, value = npu_config.set_current_run_dtype([q, k, v])
                hidden_states = npu_config.run_attention(query, key, value, atten_mask=None, input_layout="BSH",
                                                            head_dim=c, head_num=1)

                attn_output = npu_config.restore_dtype(hidden_states)

        attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        h_ = self.proj_out(attn_output)

        return x + h_
