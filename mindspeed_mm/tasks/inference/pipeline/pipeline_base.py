from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline


class MMPipeline(DiffusionPipeline):

    def prepare_latents(self, shape, generator, device, dtype, latents=None):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents, **kwargs):
        video = self.vae.decode(latents, **kwargs)  # b c t h w
        return video
