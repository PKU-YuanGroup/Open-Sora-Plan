from ..configuration_videobase import VideoBaseConfiguration
from typing import Union, Tuple

class CausalVAEConfiguration(VideoBaseConfiguration):
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size: int = 128,
        z_channels: int = 16,
        ch_mult: Tuple[int] = (1,1,2,2,4),
        num_res_block: int = 2,
        attn_resolutions: Tuple[int] = [16],
        dropout: float = 0.0,
        resolution: int = 256,
        attn_type: str = "vanilla3D",
        use_linear_attn: bool = False,
        embed_dim: int = 16,
        time_compress: int = 2,
        # ---- for loss ----
        logvar_init: float = 0.,
        kl_weight: float = 1e-6,
        pixelloss_weight: int = 1,
        perceptual_weight: int = 1, 
        disc_loss: str = "hinge",
        **kwargs
    ) -> None:
        
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.ch_mult = ch_mult
        self.num_res_block = num_res_block
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resolution = resolution
        self.attn_type = attn_type
        self.use_linear_attn = use_linear_attn
        self.embed_dim = embed_dim
        self.time_compress = time_compress

        # ---- for loss ----
        self.logvar_init = logvar_init
        self.kl_weight = kl_weight
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.disc_loss = disc_loss