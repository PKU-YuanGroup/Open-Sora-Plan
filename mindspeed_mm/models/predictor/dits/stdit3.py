import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from megatron.core import mpu
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.checkpoint import auto_grad_checkpoint
from mindspeed_mm.models.common.communications import (
    gather_forward_split_backward,
    split_forward_gather_backward,
    all_to_all,
    cal_split_sizes
)
from mindspeed_mm.models.common.attention import (
    Attention,
    MultiHeadCrossAttention,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
)
from mindspeed_mm.models.common.blocks import (
    T2IFinalLayer,
    t2i_modulate,
)
from mindspeed_mm.models.common.embeddings import (
    SizeEmbedder,
    PositionEmbedding2D,
    TimestepEmbedder,
    CaptionEmbedder,
    PatchEmbed3D,
)
from mindspeed_mm.models.common.embeddings.pos_embeddings import NpuRotaryEmbedding


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        try:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        except TypeError:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)  


class STDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flashattn=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        attn_cls = Attention
        mha_cls = MultiHeadCrossAttention

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            norm_layer=LlamaRMSNorm,
            rope=rope,
            enable_flashattn=enable_flashattn,
        )
        self.cross_attn = mha_cls(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)


    def t_mask_select(self, video_mask, x, masked_x):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # video_mask: [B, (T, S), C]
        x = torch.lerp(masked_x, x, video_mask)
        return x

    def forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        video_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if video_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                    self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)

        # modulate (attention)
        x_norm1 = self.norm1(x)
        x_m = t2i_modulate(x_norm1, shift_msa, scale_msa)
        if video_mask is not None:
            x_m_zero = t2i_modulate(x_norm1, shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(video_mask, x_m, x_m_zero)

        # attention
        if self.temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (attention)
        if video_mask is not None:
            x_m_s = self.t_mask_select(video_mask, gate_msa, gate_msa_zero) * x_m
        else:
            x_m_s = gate_msa * x_m

            # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        x = x + self.cross_attn(x, y, mask)

        # modulate (MLP)
        x_norm2 = self.norm2(x)
        x_m = t2i_modulate(x_norm2, shift_mlp, scale_mlp)
        if video_mask is not None:
            x_m_zero = t2i_modulate(x_norm2, shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(video_mask, x_m, x_m_zero)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        if video_mask is not None:
            x_m_s = self.t_mask_select(video_mask, gate_mlp, gate_mlp_zero) * x_m
        else:
            x_m_s = gate_mlp * x_m

        # residual
        x = x + self.drop_path(x_m_s)

        return x


class STDiT3(MultiModalModule):
    def __init__(
        self,
        input_sq_size=512,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flashattn=True,
        enable_sequence_parallelism=False,
        only_train_temporal=False,
        freeze_y_embedder=True,
        skip_y_embedder=False,
        **kwargs):
        super().__init__(config=None)
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels

        # model size related
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # computation related
        self.drop_path = drop_path
        self.enable_flashattn = enable_flashattn
        self.sp_size = mpu.get_context_parallel_world_size()
        self.enable_sequence_parallelism = enable_sequence_parallelism and self.sp_size > 1

        # input size related
        self.patch_size = patch_size
        self.input_sq_size = input_sq_size
        self.pos_embed = PositionEmbedding2D(hidden_size)
        self.rope = NpuRotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.fps_embedder = SizeEmbedder(hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        self.skip_y_embedder = skip_y_embedder

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, self.depth)]
        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=qk_norm,
                    enable_flashattn=enable_flashattn,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                )
                for i in range(depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, depth)]
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=qk_norm,
                    enable_flashattn=enable_flashattn,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                )
                for i in range(depth)
            ]
        )

        # final layer
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

        self.initialize_weights()
        if only_train_temporal:
            for param in self.parameters():
                param.requires_grad = False
            for block in self.temporal_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        if freeze_y_embedder:
            for param in self.y_embedder.parameters():
                param.requires_grad = False

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize timporal blocks
        for block in self.temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    def forward(self, video, timestep, prompt, prompt_mask=None, video_mask=None, fps=None, height=None, width=None, **kwargs):
        dtype = self.x_embedder.proj.weight.dtype
        B = video.size(0)
        video = video.to(dtype)
        timestep = timestep.to(dtype)
        prompt = prompt.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = video.size()
        T, H, W = self.get_dynamic_size(video)
        S = H * W
        base_size = round(S ** 0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(video, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=video.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if video_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=video.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.skip_y_embedder:
            y_lens = prompt_mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            prompt, y_lens = self.encode_text(prompt, prompt_mask)

        # === get x embed ===
        video = self.x_embedder(video)  # [B, N, C]
        video = rearrange(video, "B (T S) C -> B T S C", T=T, S=S)
        video = video + pos_emb


        # === process video mask ===
        if video_mask is not None:
            video_mask = video_mask[:, :, None, None].expand(B, T, S, video.shape[-1]).contiguous()


        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            s_split_sizes = cal_split_sizes(dim_size=video.size(2), world_size=self.sp_size)
            t_split_sizes = cal_split_sizes(dim_size=video.size(1), world_size=self.sp_size)
            video = split_forward_gather_backward(video, mpu.get_context_parallel_group(), 
                                                    dim=1, grad_scale="down", split_sizes=t_split_sizes)
            sp_rank = mpu.get_context_parallel_rank()
            if video_mask is not None:
                video_mask_split_s = video_mask[:, :, sum(s_split_sizes[:sp_rank]): sum(s_split_sizes[:sp_rank + 1]), :]
                video_mask_split_t = video_mask[:, sum(t_split_sizes[:sp_rank]): sum(t_split_sizes[:sp_rank + 1]), :, :]
                video_mask_split_s = video_mask_split_s.view(B, -1, video.shape[-1]).to(video.dtype)
                video_mask_split_t = video_mask_split_t.view(B, -1, video.shape[-1]).to(video.dtype)
            else:
                video_mask_split_s, video_mask_split_t = None, None

        T, S = video.size(1), video.size(2)
        if video_mask is not None:
            video_mask = video_mask.view(B, -1, video.shape[-1]).to(video.dtype)
        video = rearrange(video, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===
        for i, (spatial_block, temporal_block) in enumerate(zip(self.spatial_blocks, self.temporal_blocks)):
            if self.enable_sequence_parallelism:
                # === spatial block ===
                video = auto_grad_checkpoint(spatial_block, video, prompt, t_mlp, y_lens, video_mask_split_t, t0_mlp, T, S)

                # split T, gather S
                video = rearrange(video, "B (T S) C -> B T S C", T=T, S=S)
                video = all_to_all(video, mpu.get_context_parallel_group(),
                                    scatter_dim=2, scatter_sizes=s_split_sizes,
                                    gather_dim=1, gather_sizes=t_split_sizes)
                T, S = video.size(1), video.size(2)
                video = rearrange(video, "B T S C -> B (T S) C", T=T, S=S)
                
                # === temporal block ===
                video = auto_grad_checkpoint(temporal_block, video, prompt, t_mlp, y_lens, video_mask_split_s, t0_mlp, T, S)

                if i == self.depth - 1:
                    #final block
                    break
                else:
                    # split s, gather t
                    video = rearrange(video, "B (T S) C -> B T S C", T=T, S=S)
                    video = all_to_all(video, mpu.get_context_parallel_group(),
                            scatter_dim=1, scatter_sizes=t_split_sizes,
                            gather_dim=2, gather_sizes=s_split_sizes)
                    T, S = video.size(1), video.size(2)
                    video = rearrange(video, "B T S C -> B (T S) C", T=T, S=S)

            else:
                video = auto_grad_checkpoint(spatial_block, video, prompt, t_mlp, y_lens, video_mask, t0_mlp, T, S)
                video = auto_grad_checkpoint(temporal_block, video, prompt, t_mlp, y_lens, video_mask, t0_mlp, T, S)

        if self.enable_sequence_parallelism:
            # === final layer ===
            video = self.final_layer(video, t, video_mask_split_s, t0, T, S)
            video = rearrange(video, "B (T S) C -> B T S C", T=T, S=S)
            video = gather_forward_split_backward(video, mpu.get_context_parallel_group(), 
                                                    dim=2, grad_scale="up", gather_sizes=s_split_sizes)
            S = video.size(2)
            video = rearrange(video, "B T S C -> B (T S) C", T=T, S=S)
        else:
            # === final layer ===
            video = self.final_layer(video, t, video_mask, t0, T, S)
        video = self.unpatchify(video, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        video = video.to(torch.float32)
        return video

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x