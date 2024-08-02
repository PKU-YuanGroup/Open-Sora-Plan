import math
import torch
import torch.nn as nn
import torch_npu

# Todo: 后续使用megatron通信接口替换
from communications import (
    all_to_all,
    split_forward_gather_backward,
)
from megatron.core import mpu


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise AssertionError(
                "dim (%d) must be divisible by num_heads (%d)" % (dim, num_heads)
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        if self.enable_flashattn:
            qkv_permute_shape = (2, 0, 1, 3, 4)
        else:
            qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flashattn and q.dtype in [torch.float16, torch.bfloat16]:
            x = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                self.num_heads,
                input_layout="BSND",
                pse=None,
                scale=self.scale,
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0 - self.attn_drop.p if self.training else 1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not self.enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            enable_flashattn=enable_flashattn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = (
            x.shape
        )  # for sequence parallel here, the N is a local sequence length
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape)

        # sp_group = get_sequence_parallel_group()
        sp_group = mpu.get_context_parallel_group()

        # apply all_to_all to gather sequence and split attention heads
        # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

        if self.enable_flashattn:
            # [3, B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
            qkv_permute_shape = (2, 0, 1, 3, 4)
        else:
            # [3, B, NUM_HEAD_PER_DEVICE, N, HEAD_DIM]
            qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.permute(qkv_permute_shape)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flashattn and q.dtype in [torch.float16, torch.bfloat16]:
            x = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                q.shape[-2],
                input_layout="BSND",
                pse=None,
                scale=self.scale,
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0 - self.attn_drop.p if self.training else 1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        if not self.enable_flashattn:
            x = x.transpose(1, 2)

        # apply all to all to gather back attention heads and split sequence
        # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # reshape outputs back to [B, N, C]
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise AssertionError(
                "d_model (%d) must be divisible by num_heads (%d)"
                % (d_model, num_heads)
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        if x.dtype not in [torch.float16, torch.bfloat16]:
            raise AssertionError("QKV's dtype must be in bf16 or fp16")
        q = self.q_linear(x).view(-1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)

        actual_seq_qlen = []
        actual_seq_kvlen = []
        if mask is not None:
            ans = 0
            for _ in range(B):
                ans += N
                actual_seq_qlen.append(ans)
            ans = 0
            for m in mask:
                ans += m
                actual_seq_kvlen.append(ans)
        x = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            self.num_heads,
            input_layout="TND",
            pse=None,
            scale=1.0 / math.sqrt(self.head_dim),
            pre_tockens=65536,
            next_tockens=65536,
            actual_seq_qlen=tuple(actual_seq_qlen),
            actual_seq_kvlen=tuple(actual_seq_kvlen),
            keep_prob=1.0 - self.attn_drop.p,
            sparse_mode=0,
        )[0]

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelMultiHeadCrossAttention(MultiHeadCrossAttention):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        sp_group = mpu.get_context_parallel_group()
        sp_size = mpu.get_context_parallel_world_size()
        B, SUB_N, C = x.shape
        N = SUB_N * sp_size

        # shape:
        # q, k, v: [B, SUB_N, NUM_HEADS, HEAD_DIM]
        q = self.q_linear(x).view(B, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        # apply all_to_all to gather sequence and split attention heads
        q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)

        k = split_forward_gather_backward(
            k, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
        )
        v = split_forward_gather_backward(
            v, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
        )

        if x.dtype not in [torch.float16, torch.bfloat16]:
            raise AssertionError("QKV's dtype must be in bf16 or fp16")
        q = q.view(-1, self.num_heads // sp_size, self.head_dim)
        k = k.view(-1, self.num_heads // sp_size, self.head_dim)
        v = v.view(-1, self.num_heads // sp_size, self.head_dim)

        actual_seq_qlen = []
        actual_seq_kvlen = []
        if mask is not None:
            ans = 0
            for _ in range(B):
                ans += N
                actual_seq_qlen.append(ans)
            ans = 0
            for m in mask:
                ans += m
                actual_seq_kvlen.append(ans)
        x = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[-2],
            input_layout="TND",
            pse=None,
            scale=1.0 / math.sqrt(self.head_dim),
            pre_tockens=65536,
            next_tockens=65536,
            actual_seq_qlen=tuple(actual_seq_qlen),
            actual_seq_kvlen=tuple(actual_seq_kvlen),
            keep_prob=1.0 - self.attn_drop.p,
            sparse_mode=0,
        )[0]

        # apply all to all to gather back attention heads and scatter sequence
        x = x.view(B, -1, self.num_heads // sp_size, self.head_dim)
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # apply output projection
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
