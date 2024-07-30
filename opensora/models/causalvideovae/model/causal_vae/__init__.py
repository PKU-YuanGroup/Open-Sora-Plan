from .modeling_causalvae import CausalVAEModel

from einops import rearrange
from torch import nn

class CausalVAEModelWrapper(nn.Module):
    def __init__(self, model_path, subfolder=None, cache_dir=None, use_ema=False, **kwargs):
        super(CausalVAEModelWrapper, self).__init__()
        # if os.path.exists(ckpt):
        # self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
        self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs)
        if use_ema:
            self.vae.init_from_ema(model_path)
            self.vae = self.vae.ema
    def encode(self, x):  # b c t h w
        # x = self.vae.encode(x).sample()
        x = self.vae.encode(x).sample().mul_(0.18215)
        return x
    def decode(self, x):
        # x = self.vae.decode(x)
        x = self.vae.decode(x / 0.18215)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x

    def dtype(self):
        return self.vae.dtype
    #
    # def device(self):
    #     return self.vae.device