import torch
import math
import numpy as np

def get_2d_sincos_pos_embed_from_RoPE(x):
    """
    - x: (batch_size, seq_len, embed_dim)或(batch_size, head, seq_len, embed_dim)
    """
    split_dim = x.shape[-1] // 2
    emb_h, emb_w = torch.split(x, split_dim, dim=-1)
    emb_h = get_1d_sincos_pos_embed_from_RoPE(emb_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_RoPE(emb_w)  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_RoPE(x):
    """
    - x: (batch_size, seq_len, embed_dim)或(batch_size, head, seq_len, embed_dim)
    """
    if x.dim() == 4:  # 多头注意力
        batch_size, nums_head, seq_len, embed_dim = x.shape
    elif x.dim() == 3:  # 一般
        batch_size, seq_len, embed_dim = x.shape
        nums_head = None
    else:
        raise ValueError("Unsupported input shape")
    
    # 提取sincos分量
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, seq_len, embed_dim)
    sin, cos = pos_emb[..., ::2], pos_emb[..., 1::2]
    
    # 应用旋转
    if x.dim() == 4:  # 多头注意力
        cos_pos = cos.repeat_interleave(2, dim=-1)
        sin_pos = sin.repeat_interleave(2, dim=-1)
        
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape)
        x = x * cos_pos + x2 * sin_pos
    elif x.dim() == 3:  # 一般
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        x_rotated_even = cos * x_even + sin * x_odd
        x_rotated_odd = -sin * x_even + cos * x_odd
        
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2], x_rotated[..., 1::2] = x_rotated_even, x_rotated_odd
        x = x_rotated
    
    return x

def sinusoidal_position_embedding(batch_size=None, nums_head=None, seq_len=None, embed_dim=None):
    if nums_head is not None:
        # (batch_size, head, seq_len, embed_dim)
        # (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        # (embed_dim//2)
        ids = torch.arange(0, embed_dim // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / embed_dim)
        # (seq_len, embed_dim//2)
        embeddings = position * theta
        # (seq_len, embed_dim//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # (bs, head, seq_len, embed_dim//2, 2)
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))
        # (bs, head, seq_len, embed_dim)
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, seq_len, embed_dim))
    else:
        # (batch_size, seq_len, embed_dim)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(seq_len, embed_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

    return embeddings

# q = torch.ones((1, 12, 10, 32))
# q1 = get_2d_sincos_pos_embed_from_RoPE(q)
# import pdb;pdb.set_trace()
