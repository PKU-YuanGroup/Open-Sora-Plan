from ..modeling_videobase import VideoBaseAE
import torch
from torch import nn, Tensor
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import math
import os
import json
from typing import Tuple, Dict, Union
from .configuration_vqvae import VQVAEConfiguration


# Copied from https://github.com/wilson1yan/VideoGPT
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


# Copied from https://github.com/wilson1yan/VideoGPT
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim
    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims
    dims = list(range(n_dims))
    del dims[src_dim]
    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# Copied from https://github.com/wilson1yan/VideoGPT
def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0.0, training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v)  # b x n_head x d1 x ... x dn x d

    return a


# Copied from https://github.com/wilson1yan/VideoGPT
class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(
            shape=(0,) * 3,
            dim_q=n_hiddens,
            dim_kv=n_hiddens,
            n_head=n_head,
            n_layer=1,
            causal=False,
            attn_type="axial",
        )
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2), **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3), **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4), **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


# Copied from https://github.com/wilson1yan/VideoGPT
class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2),
        )

    def forward(self, x):
        return x + self.block(x)


# Copied from https://github.com/wilson1yan/VideoGPT
class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings


# Copied from https://github.com/wilson1yan/VideoGPT
class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h


# Copied from https://github.com/wilson1yan/VideoGPT
class MultiHeadAttention(nn.Module):
    def __init__(
        self, shape, dim_q, dim_kv, n_head, n_layer, causal, attn_type, attn_kwargs
    ):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False)  # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False)  # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False)  # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True)  # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == "full":
            self.attn = FullAttention(shape, causal, **attn_kwargs)
        elif attn_type == "axial":
            assert not causal, "causal axial attention is not supported"
            self.attn = AxialAttention(len(shape), **attn_kwargs)
        elif attn_type == "sparse":
            self.attn = SparseAttention(shape, n_head, causal, **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None

        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(
                        k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                        v=torch.zeros(v_shape, dtype=v.dtype, device=q.device),
                    )
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (
                    slice(None, None),
                    slice(None, None),
                    *[slice(i, i + 1) for i in decode_idx],
                )
                self.cache["k"][idx] = k
                self.cache["v"][idx] = v
            k, v = self.cache["k"], self.cache["v"]

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a)  # (b x seq_len x embd_dim)

        return a


# Copied from https://github.com/wilson1yan/VideoGPT
class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4, stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h


# Copied from https://github.com/wilson1yan/VideoGPT
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


# Copied from https://github.com/wilson1yan/VideoGPT
class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


# Copied from https://github.com/wilson1yan/VideoGPT
class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, attn_dropout=self.attn_dropout, training=self.training
        )

        return view_range(out, 2, 3, old_shape)


# Copied from https://github.com/wilson1yan/VideoGPT
class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2  # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


# Copied from https://github.com/wilson1yan/VideoGPT
class StridedSparsityConfig(object):
    """
    Strided Sparse configuration specified in https://arxiv.org/abs/1904.10509 that
    generalizes to arbitrary dimensions
    """

    def __init__(self, shape, n_head, causal, block, num_local_blocks):
        self.n_head = n_head
        self.shape = shape
        self.causal = causal
        self.block = block
        self.num_local_blocks = num_local_blocks

        assert self.num_local_blocks >= 1, "Must have at least 1 local block"
        assert self.seq_len % self.block == 0, "seq len must be divisible by block size"

        self._block_shape = self._compute_block_shape()
        self._block_shape_cum = self._block_shape_cum_sizes()

    @property
    def seq_len(self):
        return np.prod(self.shape)

    @property
    def num_blocks(self):
        return self.seq_len // self.block

    def set_local_layout(self, layout):
        num_blocks = self.num_blocks
        for row in range(0, num_blocks):
            end = min(row + self.num_local_blocks, num_blocks)
            for col in range(
                max(0, row - self.num_local_blocks), (row + 1 if self.causal else end)
            ):
                layout[:, row, col] = 1
        return layout

    def set_global_layout(self, layout):
        num_blocks = self.num_blocks
        n_dim = len(self._block_shape)
        for row in range(num_blocks):
            assert self._to_flattened_idx(self._to_unflattened_idx(row)) == row
            cur_idx = self._to_unflattened_idx(row)
            # no strided attention over last dim
            for d in range(n_dim - 1):
                end = self._block_shape[d]
                for i in range(0, (cur_idx[d] + 1 if self.causal else end)):
                    new_idx = list(cur_idx)
                    new_idx[d] = i
                    new_idx = tuple(new_idx)

                    col = self._to_flattened_idx(new_idx)
                    layout[:, row, col] = 1

        return layout

    def make_layout(self):
        layout = torch.zeros(
            (self.n_head, self.num_blocks, self.num_blocks), dtype=torch.int64
        )
        layout = self.set_local_layout(layout)
        layout = self.set_global_layout(layout)
        return layout

    def make_sparse_attn_mask(self):
        block_layout = self.make_layout()
        assert block_layout.shape[1] == block_layout.shape[2] == self.num_blocks

        num_dense_blocks = block_layout.sum().item()
        attn_mask = torch.ones(num_dense_blocks, self.block, self.block)
        counter = 0
        for h in range(self.n_head):
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    elem = block_layout[h, i, j].item()
                    if elem == 1:
                        assert i >= j
                        if i == j:  # need to mask within block on diagonals
                            attn_mask[counter] = torch.tril(attn_mask[counter])
                        counter += 1
        assert counter == num_dense_blocks

        return attn_mask.unsqueeze(0)

    def get_non_block_layout_row(self, block_layout, row):
        block_row = row // self.block
        block_row = block_layout[:, [block_row]]  # n_head x 1 x n_blocks
        block_row = block_row.repeat_interleave(self.block, dim=-1)
        block_row[:, :, row + 1 :] = 0.0
        return block_row

    ############# Helper functions ##########################

    def _compute_block_shape(self):
        n_dim = len(self.shape)
        cum_prod = 1
        for i in range(n_dim - 1, -1, -1):
            cum_prod *= self.shape[i]
            if cum_prod > self.block:
                break
        assert cum_prod % self.block == 0
        new_shape = (*self.shape[:i], cum_prod // self.block)

        assert np.prod(new_shape) == np.prod(self.shape) // self.block

        return new_shape

    def _block_shape_cum_sizes(self):
        bs = np.flip(np.array(self._block_shape))
        return tuple(np.flip(np.cumprod(bs)[:-1])) + (1,)

    def _to_flattened_idx(self, idx):
        assert len(idx) == len(
            self._block_shape
        ), f"{len(idx)} != {len(self._block_shape)}"
        flat_idx = 0
        for i in range(len(self._block_shape)):
            flat_idx += idx[i] * self._block_shape_cum[i]
        return flat_idx

    def _to_unflattened_idx(self, flat_idx):
        assert flat_idx < np.prod(self._block_shape)
        idx = []
        for i in range(len(self._block_shape)):
            idx.append(flat_idx // self._block_shape_cum[i])
            flat_idx %= self._block_shape_cum[i]
        return tuple(idx)


# Copied from https://github.com/wilson1yan/VideoGPT
class SparseAttention(nn.Module):
    ops = dict()
    attn_mask = dict()
    block_layout = dict()

    def __init__(
        self, shape, n_head, causal, num_local_blocks=4, block=32, attn_dropout=0.0
    ):  # does not use attn_dropout
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.sparsity_config = StridedSparsityConfig(
            shape=shape,
            n_head=n_head,
            causal=causal,
            block=block,
            num_local_blocks=num_local_blocks,
        )

        if self.shape not in SparseAttention.block_layout:
            SparseAttention.block_layout[self.shape] = (
                self.sparsity_config.make_layout()
            )
        if causal and self.shape not in SparseAttention.attn_mask:
            SparseAttention.attn_mask[self.shape] = (
                self.sparsity_config.make_sparse_attn_mask()
            )

    def get_ops(self):
        try:
            from deepspeed.ops.sparse_attention import MatMul, Softmax
        except:
            raise Exception(
                "Error importing deepspeed. Please install using `DS_BUILD_SPARSE_ATTN=1 pip install deepspeed`"
            )
        if self.shape not in SparseAttention.ops:
            sparsity_layout = self.sparsity_config.make_layout()
            sparse_dot_sdd_nt = MatMul(
                sparsity_layout,
                self.sparsity_config.block,
                "sdd",
                trans_a=False,
                trans_b=True,
            )

            sparse_dot_dsd_nn = MatMul(
                sparsity_layout,
                self.sparsity_config.block,
                "dsd",
                trans_a=False,
                trans_b=False,
            )

            sparse_softmax = Softmax(sparsity_layout, self.sparsity_config.block)

            SparseAttention.ops[self.shape] = (
                sparse_dot_sdd_nt,
                sparse_dot_dsd_nn,
                sparse_softmax,
            )
        return SparseAttention.ops[self.shape]

    def forward(self, q, k, v, decode_step, decode_idx):
        if self.training and self.shape not in SparseAttention.ops:
            self.get_ops()

        SparseAttention.block_layout[self.shape] = SparseAttention.block_layout[
            self.shape
        ].to(q)
        if self.causal:
            SparseAttention.attn_mask[self.shape] = (
                SparseAttention.attn_mask[self.shape].to(q).type_as(q)
            )
        attn_mask = SparseAttention.attn_mask[self.shape] if self.causal else None

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        if decode_step is not None:
            mask = self.sparsity_config.get_non_block_layout_row(
                SparseAttention.block_layout[self.shape], decode_step
            )
            out = scaled_dot_product_attention(
                q, k, v, mask=mask, training=self.training
            )
        else:
            if q.shape != k.shape or k.shape != v.shape:
                raise Exception("SparseAttention only support self-attention")
            sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops()
            scaling = float(q.shape[-1]) ** -0.5

            attn_output_weights = sparse_dot_sdd_nt(q, k)
            if attn_mask is not None:
                attn_output_weights = attn_output_weights.masked_fill(
                    attn_mask == 0, float("-inf")
                )
            attn_output_weights = sparse_softmax(attn_output_weights, scale=scaling)

            out = sparse_dot_dsd_nn(attn_output_weights, v)

        return view_range(out, 2, 3, old_shape)


# Modified from https://github.com/wilson1yan/VideoGPT
class VQVAEModel(VideoBaseAE):

    DOWNLOADED_VQVAE = {
        "bair_stride4x2x2": "1iIAYJ2Qqrx5Q94s5eIXQYJgAydzvT_8L",
        "ucf101_stride4x4x4": "1uuB_8WzHP_bbBmfuaIV7PK_Itl3DyHY5",
        "kinetics_stride4x4x4": "1DOvOZnFAIQmux6hG7pN_HkyJZy3lXbCB",
        "kinetics_stride2x4x4": "1jvtjjtrtE4cy6pl7DK_zWFEPY3RZt2pB",
    }

    def __init__(self, config: VQVAEConfiguration):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.n_codes = config.n_codes
        self.encoder = Encoder(config.n_hiddens, config.n_res_layers, config.downsample)
        self.decoder = Decoder(config.n_hiddens, config.n_res_layers, config.downsample)
        self.pre_vq_conv = SamePadConv3d(config.n_hiddens, config.embedding_dim, 1)
        self.post_vq_conv = SamePadConv3d(config.embedding_dim, config.n_hiddens, 1)
        self.codebook = Codebook(config.n_codes, config.embedding_dim)

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        recon_loss = F.mse_loss(x_recon, x) / 0.06
        return recon_loss, x_recon, vq_output

    def encode(self, x: Tensor, include_embeddings: bool = False) -> Union[Tuple[Tensor, Tensor], Tensor]:
        h = self.pre_vq_conv(self.encoder(x))
        vq_output: Dict[str, Tensor] = self.codebook(h)
        if include_embeddings:
            return vq_output["encodings"], vq_output["embeddings"]
        else:
            return vq_output["encodings"]

    def decode(self, encodings: Tensor) -> Tensor:
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    @classmethod
    def load_from_checkpoint(cls, model_path):
        if not os.path.isdir(model_path):
            """model downloaded from internet"""
            model_cpkt = torch.load(model_path)
            # Compatible with old videogpt model formats.
            if "hyper_parameters" in model_cpkt:
                hyper_parameters = vars(model_cpkt.get("hyper_parameters").get("args"))
                state_dict = model_cpkt.get("state_dict")
                model = cls(config=VQVAEConfiguration(**hyper_parameters))
                model.load_state_dict(state_dict)
                return model
            else:
                raise RuntimeError("Model checkpoint has a wrong format.")
        else:
            with open(os.path.join(model_path, "config.json"), "r") as file:
                config = json.load(file)
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
            model = cls(config=VQVAEConfiguration(**config))
            model.load_state_dict(state_dict)
            return model
            
    @classmethod
    def download_and_load_model(cls, model_name, cache_dir=None):
        from .....utils.downloader import gdown_download
        path = gdown_download(
            cls.DOWNLOADED_VQVAE[model_name], model_name, cache_dir=cache_dir
        )
        return cls.load_from_checkpoint(path)
