import torch
from einops import rearrange

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline


class MMPipeline(DiffusionPipeline):

    def prepare_latents(self, shape, generator, device, dtype, latents=None):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents):
        print(f"before vae decode {latents.shape}",
              torch.max(latents).item(), torch.min(latents).item(),
              torch.mean(latents).item(), torch.std(latents).item())
        video = self.vae.decode(latents)  # [b, c, t, h, w]
        print(f"after vae decode {video.shape}",
              torch.max(video).item(), torch.min(latents).item(),
              torch.mean(video).item(), torch.std(latents).item())
        # [b, c, t, h, w] --> [b, t, h, w, c]
        video = rearrange(video, "b c t h w -> b t h w c").contiguous()
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu()
        return video
