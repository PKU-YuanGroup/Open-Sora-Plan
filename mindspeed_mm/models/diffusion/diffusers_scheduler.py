# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union, Tuple, List, Callable

from tqdm.auto import tqdm
import torch
from torch import Tensor


from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)

from mindspeed_mm.models.predictor import PredictModel
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
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler
}


class DiffusersScheduler:
    """
    Wrapper class for diffusers sheduler.
    Args:
        config:
        {
            "num_train_timesteps":1000,
            "beta_start":0.0001,
            "beta_end":0.02
            "beta_schedule":"linear"
            ...
        }
    """
    def __init__(self, config):
        if config.model_id in DIFFUSERS_SCHEDULE_MAPPINGS:
            model_cls = DIFFUSERS_SCHEDULE_MAPPINGS[config.model_id]
            self.diffusion = model_cls(**config)
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self.do_classifier_free_guidance = config.guidance_scale > 1.0
        self.num_timesteps = config.num_timesteps
        self.guidance_scale = config.guidance_scale
        self.noise_offset = config.noise_offset

    def training_losses(self):
        raise NotImplementedError()

    def sample(
        self,
        model: PredictModel,
        shape: Union[List, Tuple],
        latents: Tensor,
        clip_denoised: bool = True,
        denoised_fn: Callable = None,
        cond_fn: Callable = None,
        model_kwargs: dict = None,
        device: dict = None,
        progress: bool = True,
        mask: Tensor = None,
        callback=None,
        callback_steps: int = 1,
        added_cond_kwargs: dict = None,
        extra_step_kwargs: dict = None
    ) -> Tensor:
        """
        model_kwargs: dict
            {
                "attention_mask": attention_mask,
                "encoder_hidden_states": prompt_embeds
                "encoder_attention_mask": prompt_attention_mask
            }
        """
        indices = list(range(self.num_timesteps))[::-1]
        if device is None:
            device = next(model.parameters()).device
        if progress:
            indices = tqdm(indices)
        if not isinstance(shape, (tuple, list)):
            raise AssertionError("param shape is incorrect")
        if latents is None:
            latents = torch.randn(*shape, device=device)
        if self.do_classifier_free_guidance:
            latents = torch.cat([latents, latents], 0)
        
        model_kwargs.update(added_cond_kwargs)
        model_kwargs["hidden_states"] = latents
        
        # for loop denoising to get latents
        for i in indices:
            timestep = torch.tensor([i] * shape[0], device=device)

            with torch.no_grad():
                noise_pred = model(t=timestep, **model_kwargs)[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # learned sigma
            if model.out_channels // 2 == model.in_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            # compute previous image: x_t -> x_t-1
            latents = self.diffusion.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.diffusion, "order", 1)
                callback(step_idx, timestep, latents)
        
        return latents
