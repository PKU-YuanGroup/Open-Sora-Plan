from typing import Union, Tuple, List, Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.training_utils import compute_snr

from mindspeed_mm.utils.utils import get_device
DIFFUSERS_SCHEDULE_MAPPINGS = {
    "DDIM": DDIMScheduler,
    "EulerDiscrete": EulerDiscreteScheduler,
    "DDPM": DDPMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "PNDM": PNDMScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
}


class DiffusersScheduler:
    """
    Wrapper class for diffusers sheduler.
    Args:
        config:
        {
            "model_id":"PNDM"
            "num_train_steps":1000,
            "beta_start":0.0001,
            "beta_end":0.02
            "beta_schedule":"linear"
            ...
        }
    """

    def __init__(self, config):
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self.guidance_scale = config.pop("guidance_scale", 1.0)
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        self.num_train_steps = config.pop("num_train_steps", 1000)
        self.num_inference_steps = config.pop("num_inference_steps", None)
        self.prediction_type = config.pop("prediction_type", "epsilon")
        self.noise_offset = config.pop("noise_offset", 0)
        self.snr_gamma = config.pop("snr_gamma", 5.0)
        self.device = get_device(config.pop("device", "npu"))
        model_id = config.pop("model_id")

        if model_id in DIFFUSERS_SCHEDULE_MAPPINGS:
            model_cls = DIFFUSERS_SCHEDULE_MAPPINGS[model_id]
            self.diffusion = model_cls(**config)

        # Prepare timesteps for inference
        if self.num_inference_steps:
            self.diffusion.set_timesteps(self.num_inference_steps)
            self.timesteps = self.diffusion.timesteps
            self.num_warmup_steps = max(
                len(self.timesteps) - self.num_inference_steps * self.diffusion.order, 0
            )
    
    def training_losses(
        self,
        model_output: Tensor,
        x_start: Tensor,
        noise: Tensor = None,
        mask: Tensor = None,
        t: Tensor = None,
        **kwargs
        ) -> Tensor:
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.diffusion.get_velocity(x_start, noise, t)
        elif self.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = x_start
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_output = model_output - noise
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        b, c, _, _, _ = model_output.shape
        if torch.all(mask.bool()):
            mask = None
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, c, 1, 1, 1).float()  # b t h w -> b c t h w
            mask = mask.reshape(b, -1)
        if self.snr_gamma is None:
            # model_pred: [b, c, t, h, w], mask: [b, t, h, w]
            loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
            loss = loss.reshape(b, -1)
            if mask is not None:
                loss = (loss * mask).sum() / mask.sum()  # mean loss on unpad patches
            else:
                loss = loss.mean()
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.diffusion, t)
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(t)], dim=1).min(
                dim=1
            )[0]
            if self.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
            loss = loss.reshape(b, -1)
            mse_loss_weights = mse_loss_weights.reshape(b, 1)
            if mask is not None:
                loss = (loss * mask * mse_loss_weights).sum() / mask.sum()  # mean loss on unpad patches
            else:
                loss = (loss * mse_loss_weights).mean()
        return loss
    
    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor = None,
        noise: Tensor = None,
    ) -> Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        b, c, _, _, _ = x_start.shape
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError("The shape of noise and x_start must be equal.")
        if t is None:
            t = torch.randint(0, self.num_train_steps, (b,), device=x_start.device)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn((b, c, 1, 1, 1), device=x_start.device)
        x_t = self.diffusion.add_noise(x_start, noise, t)
        return x_t, noise, t

    def sample(
        self,
        model: Callable,
        shape: Union[List, Tuple],
        latents: Tensor,
        model_kwargs: dict = None,
        mask: Tensor = None,
        callback=None,
        callback_steps: int = 1,
        added_cond_kwargs: dict = None,
        extra_step_kwargs: dict = None,
        **kwargs
    ) -> Tensor:
        """
        Generate samples from the model.
        :param model: the noise predict model.
        :param shape: the shape of the samples, (N, C, H, W).
        :param latents: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
            {
                "attention_mask": attention_mask,
                "encoder_hidden_states": prompt_embeds
                "encoder_attention_mask": prompt_attention_mask
            }
        :return: a non-differentiable batch of samples.
        Returns clean latents.
        """
        if not isinstance(shape, (tuple, list)):
            raise AssertionError("param shape is incorrect")
        if latents is None:
            latents = torch.randn(*shape, device=self.device)
        if added_cond_kwargs:
            model_kwargs.update(added_cond_kwargs)

        self.diffusion.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps = self.diffusion.timesteps

        # for loop denoising to get latents
        with tqdm(total=self.num_inference_steps) as progress_bar:
            for i, t in enumerate(self.timesteps):
                # timestep = torch.tensor([i] * shape[0], device=self.device)
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.diffusion.scale_model_input(latent_model_input, t)
                current_timestep = t
                current_timestep = current_timestep.expand(latent_model_input.shape[0])
                model_kwargs["latents"] = latent_model_input
                video_mask = torch.ones_like(latent_model_input)[:, 0]
                world_size = model_kwargs.get("world_size", 1)
                video_mask = video_mask.repeat(1, world_size, 1, 1)
                model_kwargs["video_mask"] = video_mask

                with torch.no_grad():
                    noise_pred = model(timestep=current_timestep, **model_kwargs)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if model.out_channels // 2 == model.in_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.diffusion.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(self.timesteps) - 1 or (
                        (i + 1) > self.num_warmup_steps and (i + 1) % self.diffusion.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.diffusion, "order", 1)
                        callback(step_idx, t, latents)
        return latents
