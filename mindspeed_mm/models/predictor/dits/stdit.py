import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.checkpoint import auto_grad_checkpoint
from mindspeed_mm.models.common.communications import (
    gather_forward_split_backward,
    split_forward_gather_backward,
    all_to_all,
)
from mindspeed_mm.models.common.attention import (
    Attention,
    MultiHeadCrossAttention,
)
from mindspeed_mm.models.common.blocks import (
    T2IFinalLayer,
    t2i_modulate,
)
from mindspeed_mm.models.common.embeddings import (
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    TimestepEmbedder,
    CaptionEmbedder,
    PatchEmbed3D,
)


class STDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self._enable_sequence_parallelism = enable_sequence_parallelism

        self.attn_cls = Attention
        self.mha_cls = MultiHeadCrossAttention
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
        )
        self.cross_attn = self.mha_cls(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t

        if self._enable_sequence_parallelism:
            self.sp_size = mpu.get_context_parallel_world_size()
            # make sure d_t is divisible by sp_size
            if d_t % self.sp_size != 0:
                raise AssertionError(
                    "d_t (%d) must be divisible by sp_size (%d)" % (d_t, self.sp_size)
                )
            self.d_t = d_t // self.sp_size

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
        )

    def forward(self, x, y, t, mask=None, tpe=None):
        mask = mask.tolist()
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal to spatital switch in dsp
        if self._enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=self.d_t, S=self.d_s)
            x = all_to_all(
                x, mpu.get_context_parallel_group(), scatter_dim=2, gather_dim=1
            )
            self.d_t = self.d_t * self.sp_size
            self.d_s = self.d_s // self.sp_size
            x = rearrange(x, "B T S C -> B (T S) C", T=self.d_t, S=self.d_s)

        # temporal branch
        x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        if tpe is not None:
            x_t = x_t + tpe
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # spatital to temporal switch in dsp
        if self._enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=self.d_t, S=self.d_s)
            x = all_to_all(
                x, mpu.get_context_parallel_group(), scatter_dim=1, gather_dim=2
            )
            self.d_t = self.d_t // self.sp_size
            self.d_s = self.d_s * self.sp_size
            x = rearrange(x, "B T S C -> B (T S) C", T=self.d_t, S=self.d_s)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # mlp
        x = x + self.drop_path(
            gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        )

        return x


class STDiT(MultiModalModule):
    def __init__(
        self,
        input_size=(1, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        dtype=torch.float32,
        space_scale=1.0,
        time_scale=1.0,
        freeze=None,
        enable_flashattn=False,
        enable_sequence_parallelism=False,
        **kwargs,
    ):
        super().__init__(config=None)
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.space_scale = space_scale
        self.time_scale = time_scale

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        if mpu.is_pipeline_first_stage():
            self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
            self.t_embedder = TimestepEmbedder(hidden_size)
            self.t_block = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
            self.y_embedder = CaptionEmbedder(
                in_channels=caption_channels,
                hidden_size=hidden_size,
                uncond_prob=class_dropout_prob,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                token_num=model_max_length,
            )
        else:
            self.x_embedder = None
            self.t_embedder = None
            self.t_block = None
            self.y_embedder = None

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.num_layers = self._get_num_layers(depth)

        args = get_args()
        if args.virtual_pipeline_model_parallel_size is not None:
            raise NotImplementedError("VPP is not supported now")
        else:
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers
        enable_sequence_parallelism = (
            enable_sequence_parallelism and mpu.get_context_parallel_world_size() > 1
        )
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        if self.recompute_granularity == "selective":
            raise ValueError(
                "recompute_granularity does not support selective mode in STDiT"
            )
        if self.distribute_saved_activations:
            raise NotImplementedError(
                "distribute_saved_activations is currently not supported"
            )

        self.blocks = nn.ModuleList(
            [
                self.build_layer(drop_path, i + offset, enable_sequence_parallelism)
                for i in range(self.num_layers)
            ]
        )

        if mpu.is_pipeline_last_stage():
            self.final_layer = T2IFinalLayer(
                hidden_size, np.prod(self.patch_size), self.out_channels
            )
        else:
            self.final_layer = None

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if freeze is not None:
            if freeze not in ["not_temporal", "text"]:
                raise AssertionError
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()

        # sequence parallel related configs
        self.enable_sequence_parallelism = enable_sequence_parallelism
        if enable_sequence_parallelism:
            self.sp_rank = mpu.get_context_parallel_rank()
        else:
            self.sp_rank = None

    def _get_layer(self, layer_number):
        return self.blocks[layer_number]

    def _checkpointed_forward(self, x, y, t0, y_lens, tpe):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_

            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_num = 0
            while layer_num < self.num_layers:
                if layer_num == 0:
                    tpe = self.pos_embed_temporal
                else:
                    tpe = None
                x = tensor_parallel.checkpoint(
                    custom(layer_num, layer_num + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    x,
                    y,
                    t0,
                    y_lens,
                    tpe,
                )
                layer_num += self.recompute_num_layers
        elif self.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for layer_num in range(self.num_layers):
                if layer_num == 0:
                    tpe = self.pos_embed_temporal
                else:
                    tpe = None

                if layer_num < self.recompute_num_layers:
                    x = tensor_parallel.checkpoint(
                        custom(layer_num, layer_num + 1),
                        self.distribute_saved_activations,
                        x,
                        y,
                        t0,
                        y_lens,
                        tpe,
                    )
                else:
                    block = self._get_layer(layer_num)
                    x = block(x, y, t0, y_lens, tpe)
        else:
            raise ValueError("Invalid activation recompute method.")
        return x

    def build_layer(self, drop_path, layer_number, enable_sequence_parallelism):
        return STDiTBlock(
            self.hidden_size,
            self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_path=drop_path[layer_number],
            enable_flashattn=self.enable_flashattn,
            enable_sequence_parallelism=enable_sequence_parallelism,
            d_t=self.num_temporal,
            d_s=self.num_spatial,
        )

    def forward(self, video, timestep, prompt, prompt_mask=None, **kwargs):
        """
        Forward pass of STDiT.
        Args:
            video (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            prompt (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            prompt_mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        if mpu.is_pipeline_first_stage():
            x = video.to(self.dtype)
            timestep = timestep.to(self.dtype)
            y = prompt.to(self.dtype)
            mask = prompt_mask

            # embedding
            x = self.x_embedder(x)  # [B, N, C]
            x = rearrange(
                x, "B (T S) C -> B T S C", T=self.num_temporal, S=self.num_spatial
            )
            x = x + self.pos_embed
            x = rearrange(x, "B T S C -> B (T S) C")

            # shard over the sequence dim if sp is enabled
            if self.enable_sequence_parallelism:
                x = split_forward_gather_backward(
                    x, mpu.get_context_parallel_group(), dim=1, grad_scale="down"
                )

            t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
            t0 = self.t_block(t)  # [B, C]
            y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

            if mask is not None:
                if mask.shape[0] != y.shape[0]:
                    mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
                mask = mask.squeeze(1).squeeze(1)
                y = (
                    y.squeeze(1)
                    .masked_select(mask.unsqueeze(-1) != 0)
                    .view(1, -1, x.shape[-1])
                )
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = [y.shape[2]] * y.shape[0]
                y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            x, y, t0, y_lens, tpe = self.input_tensor

        # blocks
        tpe = None
        y_lens = torch.tensor(y_lens)
        if self.recompute_granularity == "full":
            x = self._checkpointed_forward(x, y, t0, y_lens, tpe)
        else:
            for i, block in enumerate(self.blocks):
                if i == 0:
                    tpe = self.pos_embed_temporal
                else:
                    tpe = None
                x = block(x, y, t0, y_lens, tpe)

        if mpu.is_pipeline_last_stage():
            if self.enable_sequence_parallelism:
                x = gather_forward_split_backward(
                    x, mpu.get_context_parallel_group(), dim=1, grad_scale="up"
                )
                # x.shape: [B, N, C]

            # final process
            x = self.final_layer(x, t)  # [B, N, C=T_p * H_p * W_p * C_out]
            x = self.unpatchify(x)  # [B, C_out, T, H, W]

            # cast to float32 for better accuracy
            x = x.to(torch.float32)
            return x
        else:
            # for next stage in pipeline parallel
            # x: output latent representation
            # y: representation of prompts
            # t0: representation of timestep
            # y_lens: lengths of prompts
            # tpe: position embedding of temporal dimension
            return x, y, t0, y_lens, tpe

    def unpatchify(self, x):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
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
        return x

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            interpolation_scale=(self.space_scale, self.space_scale),
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            interpolation_scale=self.time_scale,
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
