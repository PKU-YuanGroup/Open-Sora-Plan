from tqdm.auto import tqdm
import torch
from torch import Tensor
from torch.distributions import LogisticNormal

from .diffusion_utils import extract_into_tensor, mean_flat


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t
    
    
class RFlow:
    def __init__(
        self,
        num_inference_steps: int = None,
        num_train_steps=1000,
        use_discrete_timesteps=False,
        use_timestep_transform=True,
        sample_method="logit-normal",
        cfg_scale=4.0,
        loc=0.0,
        scale=1.0,
        transform_scale=1.0,
        **kwargs,
    ):
        self.num_sampling_steps = num_inference_steps
        self.num_timesteps = num_train_steps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        if sample_method not in ["uniform", "logit-normal"]:
            raise Exception("Currently sample_method must be uniform or logit-normal")

        if use_discrete_timesteps and not sample_method == "uniform":
            raise Exception("Only uniform sampling is supported for discrete timesteps")

        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

    def sample(
        self,
        model,
        latents,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        model_kwargs=None,
        progress=True,
        **kwargs
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        if additional_args is not None:
            model_kwargs.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * latents.shape[0], device=latents.device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, model_kwargs, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i in progress_wrap(range(len(timesteps))):
            t = timesteps[i]
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = latents.clone()
                x_noise, _, _ = self.q_sample(x_start=x0, noise=torch.randn_like(x0), t=t)
                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_kwargs["video_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                latents = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([latents, latents], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_kwargs).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            latents = latents + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                latents = torch.where(mask_t_upper[:, None, :, None, None], latents, x0)

        return latents
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise

    def q_sample(self, x_start, noise=None, t=None, mask=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs=model_kwargs, scale=self.transform_scale,
                                       num_timesteps=self.num_timesteps)

        if noise is None:
            noise = torch.randn_like(x_start)
        if not noise.shape == x_start.shape:
            raise Exception("noise must have the same shape as x_start")
        
        x_t = self.add_noise(x_start, noise, t)
        
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)
        return x_t, noise, t

    def training_losses(
        self, 
        model_output, 
        x_start, 
        noise=None, 
        t=None,
        mask=None, 
        weights=None, 
        **kwargs) -> Tensor:
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        velocity_pred = model_output.chunk(2, dim=1)[0]
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)

        return loss
