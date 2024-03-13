import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed
import torch.distributed as dist
from torch.distributed import ProcessGroup

from opensora.models.diffusion.mmdit.common.util.operation import AllGather, AsyncAllGatherForTwo, all_to_all_comm

# TODO: make attention use processors similar to attn.py
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

try:
    # needs to have https://github.com/corl-team/rebased/ installed
    from fla.ops.triton.rebased_fast import parallel_rebased
except:
    REBASED_IS_AVAILABLE = False

try:
    # needs to have https://github.com/lucidrains/ring-attention-pytorch installed
    from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda
except:
    RING_ATTENTION_IS_AVAILABLE = False

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class JointAttention(nn.Module):
    def __init__(self, txt_dim, pix_dim,
                        num_heads=8,
                        qkv_bias=False,
                        attn_drop=0.,
                        proj_drop=0.,
                        use_lora=False,
                        attention_mode='math',
                        eps=1e-12,
                        causal=True,
                        ring_bucket_size=1024,
                        sequence_parallel_size: int = 1,
                        sequence_parallel_group: Optional[ProcessGroup] = None,
                        sequence_parallel_type: str = None,
                        sequence_parallel_overlap: bool = False,
                        sequence_parallel_overlap_size: int = 2,):
        super().__init__()
        dim = txt_dim + pix_dim
        assert txt_dim % num_heads == 0, 'dim should be divisible by num_heads'
        assert pix_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv_text = nn.Linear(txt_dim, txt_dim * 3, bias=qkv_bias)
        self.qkv_pix = nn.Linear(pix_dim, pix_dim * 3, bias=qkv_bias)
        self.rms_q_text = RMSNorm(txt_dim)
        self.rms_k_text = RMSNorm(txt_dim)
        self.rms_q_pix = RMSNorm(pix_dim)
        self.rms_k_pix = RMSNorm(pix_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_pix = nn.Linear(dim, txt_dim)
        self.proj_text = nn.Linear(dim, pix_dim)
        self.proj_drop_text = nn.Dropout(proj_drop)
        self.proj_drop_pix = nn.Dropout(proj_drop)
        self.eps = eps
        self.causal = causal
        self.ring_bucket_size = ring_bucket_size

        # sequence parallel
        self.sequence_parallel_type = sequence_parallel_type
        self.sequence_parallel_size = sequence_parallel_size
        if sequence_parallel_size == 1:
            sequence_parallel_type = None
        else:
            assert sequence_parallel_type in [
                "longseq",
                "ulysses",
            ], "sequence_parallel_type should be longseq or ulysses"
        self.sequence_parallel_group = sequence_parallel_group
        self.sequence_parallel_overlap = sequence_parallel_overlap
        self.sequence_parallel_overlap_size = sequence_parallel_overlap_size
        if self.sequence_parallel_size > 1:
            assert (
                self.num_heads % self.sequence_parallel_size == 0
            ), "num_heads should be divisible by sequence_parallel_size"
            self.sequence_parallel_rank = dist.get_rank(sequence_parallel_group)
            self.sequence_parallel_param_slice = slice(
                self.qkv.out_features // sequence_parallel_size * self.sequence_parallel_rank,
                self.qkv.out_features // sequence_parallel_size * (self.sequence_parallel_rank + 1),
            )

    def make_qkv(self, x, qkv_weights):
        if self.sequence_parallel_type == "longseq":
            if self.sequence_parallel_overlap:
                if self.sequence_parallel_size == 2:
                    # (B, N / SP_SIZE, C) => (SP_SIZE * B, N / SP_SIZE, C)
                    qkv = AsyncAllGatherForTwo.apply(
                        x,
                        qkv_weights.weight[self.sequence_parallel_param_slice],
                        qkv_weights.bias[self.sequence_parallel_param_slice],
                        self.sequence_parallel_rank,
                        self.sequence_parallel_size,
                        dist.group.WORLD,
                    )  # (B, N, C / SP_SIZE)
                else:
                    raise NotImplementedError(
                        "sequence_parallel_overlap is only supported for sequence_parallel_size=2"
                    )
            else:
                # (B, N / SP_SIZE, C) => (SP_SIZE * B, N / SP_SIZE, C)
                x = AllGather.apply(x)[0]
                # (SP_SIZE, B, N / SP_SIZE, C) => (B, N, C)
                x = rearrange(x, "sp b n c -> b (sp n) c")
                qkv = F.linear(
                    x,
                    qkv_weights.weight[self.sequence_parallel_param_slice],
                    qkv_weights.bias[self.sequence_parallel_param_slice],
                )
        else:
            qkv = qkv_weights(x)  # (B, N, C), N here is N_total // SP_SIZE
        return qkv
    
    def split_qkv(self, qkv, B, N, total_N, num_heads):
        if self.sequence_parallel_type == "ulysses":
            q, k, v = qkv.split(self.head_dim * self.num_heads, dim=-1)
            q = all_to_all_comm(q, self.sequence_parallel_group)
            k = all_to_all_comm(k, self.sequence_parallel_group)
            v = all_to_all_comm(v, self.sequence_parallel_group)

            if self.enable_flashattn:
                q = q.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim).contiguous()
                k = k.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim).contiguous()
                v = v.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim).contiguous()
            else:
                q = (
                    q.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                k = (
                    k.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                v = (
                    v.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

        else:
            if self.sequence_parallel_type == "longseq":
                qkv_shape = (B, total_N, num_heads, 3, self.head_dim)
                if self.enable_flashattn:
                    qkv_permute_shape = (3, 0, 1, 2, 4)
                else:
                    qkv_permute_shape = (3, 0, 2, 1, 4)
            else:
                qkv_shape = (B, total_N, 3, num_heads, self.head_dim)
                if self.enable_flashattn:
                    qkv_permute_shape = (2, 0, 1, 3, 4)
                else:
                    qkv_permute_shape = (2, 0, 3, 1, 4)
            qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
            q, k, v = qkv.unbind(0)

            return q, k, v

    def forward(self, x, c):
        B, N, C1 = x.shape
        total_N = N * self.sequence_parallel_size

        qkv_pix = self.make_qkv(x, self.qkv_pix)

        num_heads = (
            self.num_heads if self.sequence_parallel_type is None else self.num_heads // self.sequence_parallel_size
        )
        q_pix, k_pix, v_pix = self.split_qkv(qkv_pix, B, N, total_N, num_heads)
        q_pix = self.rms_q_pix(q_pix)
        k_pix = self.rms_k_pix(k_pix)

        # Assuming 
        B, N, C2 = c.shape
        qkv_text = self.make_qkv(c, self.qkv_text)
        # TODO: make separate num_heads for text?
        q_text, k_text, v_text = self.split_qkv(qkv_text, B, N, total_N, num_heads)

        if self.attention_mode != 'rebased':
            # Rebased does RMS norm inside already
            q_text = self.rms_q_text(q_text)
            k_text = self.rms_k_text(k_text)

        q = torch.cat([q_text, q_pix], dim=-1)
        k = torch.cat([k_text, k_pix], dim=-1)
        v = torch.cat([v_text, v_pix], dim=-1)

        C = C1 + C2

        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            z = xformers.ops.memory_efficient_attention(q, k, v)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                z = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p, scale=self.scale) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            z = (attn @ v)

        elif self.attention_mode == 'rebased':
            z = parallel_rebased(q, k, v, self.eps, True, True)

        elif self.attention_mode == 'ring':
            z = ring_flash_attn_cuda(q, k, v, causal=self.causal, bucket_size=self.ring_bucket_size)

        else:
            raise NotImplemented
        
        if self.sequence_parallel_type is None:
            z_output_shape = (B, N, C)
        else:
            z_output_shape = (B, total_N, num_heads * self.head_dim)

        if self.attention_mode == 'flash' or self.attention_mode == 'ring' or self.attention_mode == 'rebased':
            z = z.reshape(x_output_shape)
        else:
            z = z.transpose(1, 2).reshape(x_output_shape)

        if self.sequence_parallel_size > 1:
            z = all_to_all_comm(z, self.sequence_parallel_group, scatter_dim=1, gather_dim=2)

        x = self.proj_pix(z)
        x = self.proj_drop_pix(x)

        c = self.proj_text(z)
        c = self.proj_drop_text(c)

        return x, c

    # Rearrange the qkv projection (qkv ... qkv <-> q ... q k ... k v ... v)
    def rearrange_fused_weight(self, layer: nn.Linear, flag="load"):
        # check whether layer is an torch.nn.Linear layer
        if not isinstance(layer, nn.Linear):
            raise ValueError("Invalid layer type for fused qkv weight rearrange!")

        with torch.no_grad():
            if flag == "load":
                layer.weight.data = rearrange(layer.weight.data, "(x NH H) D -> (NH x H) D", x=3, H=self.head_dim)
                layer.bias.data = rearrange(layer.bias.data, "(x NH H) -> (NH x H)", x=3, H=self.head_dim)
                assert layer.weight.data.is_contiguous()
                assert layer.bias.data.is_contiguous()

            elif flag == "save":
                layer.weight.data = rearrange(layer.weight.data, "(NH x H) D -> (x NH H) D", x=3, H=self.head_dim)
                layer.bias.data = rearrange(layer.bias.data, "(NH x H) -> (x NH H)", x=3, H=self.head_dim)
                assert layer.weight.data.is_contiguous()
                assert layer.bias.data.is_contiguous()
            else:
                raise ValueError("Invalid flag for fused qkv weight rearrange!")
