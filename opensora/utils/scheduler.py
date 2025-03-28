# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class OpenSoraFlowMatchEulerSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class OpenSoraFlowMatchEulerScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        shift: float = 1.0,
        use_dynamic_shifting: Optional[bool] = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        weighting_scheme: str = 'logit_norm',
        sigma_eps: Optional[float] = None,
        rescale: float = 1000.0
    ):
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.weighting_scheme = weighting_scheme
        self.rescale = rescale

        if sigma_eps is not None:
            if not (sigma_eps >= 0 and sigma_eps <= 1e-2):
                return ValueError("sigma_eps should be in the range of [0, 1e-2]") 
        else:
            sigma_eps = 0.0

        self._sigma_eps = sigma_eps
        self._sigma_min = 0.0 
        self._sigma_max = 1.0  

        self.sigmas = None

    @property
    def sigma_eps(self):
        return self._sigma_eps

    @property
    def sigma_min(self):
        return self._sigma_min

    @property
    def sigma_max(self):
        return self._sigma_max

    def add_noise(
        self,
        sample: torch.FloatTensor,
        sigmas: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample
            sigma (`float` or `torch.FloatTensor`):
                sigma value in flow matching.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        prev_dtype = sample.dtype
        
        # Upcast to avoid precision issues when computing prev_sample
        sigmas = sigmas.to(torch.float32)
        noise = noise.to(torch.float32)
        sample = sample.to(torch.float32)

        noised_sample = sigmas * noise + (1.0 - sigmas) * sample

        # Cast sample back to model compatible dtype
        noised_sample = noised_sample.to(prev_dtype)
        return noised_sample
    
    def compute_density_for_sigma_sampling(
        self, 
        batch_size: int, 
        logit_mean: float = None, 
        logit_std: float = None, 
        mode_scale: float = None, 
    ):
        """Compute the density for sampling the sigmas when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if self.weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            sigmas = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
            sigmas = torch.nn.functional.sigmoid(sigmas)
        elif self.weighting_scheme == "mode":
            sigmas = torch.rand(size=(batch_size,), device="cpu")
            sigmas = 1 - sigmas - mode_scale * (torch.cos(math.pi * sigmas / 2) ** 2 - 1 + sigmas)
        else:
            sigmas = torch.rand(size=(batch_size,), device="cpu")

        sigmas = torch.where(sigmas >= self.sigma_eps, sigmas, torch.ones_like(sigmas) * self.sigma_eps)

        return sigmas
    
    def compute_loss_weighting_for_sd3(self, sigmas=None):
        """Computes loss weighting scheme for SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if self.weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif self.weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
        else:
            weighting = torch.ones_like(sigmas)
        return weighting

    def sigma_shift(
        self, 
        sigmas: Union[float, torch.Tensor], 
        shift: float, 
        dynamic: Optional[bool] = False, 
        mu: Optional[float] = None,
        gamma: Optional[float] = 1.0
    ):
        if not dynamic:
            sigmas_ = shift * sigmas / (1 + (shift - 1) * sigmas)
        else:
            sigmas_ = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1) ** gamma)
        if isinstance(sigmas_, torch.Tensor):
            sigmas_ = torch.where(sigmas_ > self.sigma_eps, sigmas_, torch.ones_like(sigmas_) * self.sigma_eps)
        else:
            sigmas_ = max(sigmas_, self.sigma_eps)
        return sigmas_

    def set_sigmas(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        inversion: Optional[bool] = False,
        **kwargs,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            sigmas = np.linspace(self._sigma_max, self._sigma_min, num_inference_steps + 1)

        if inversion:
            sigmas = sigmas[::-1]

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        if self.config.use_dynamic_shifting:
            sigmas = self.sigma_shift(sigmas, self.config.shift, dynamic=True, mu=mu)
        else:
            sigmas = self.sigma_shift(sigmas, self.config.shift)

        self.sigmas = sigmas

        return sigmas

    def step(
        self,
        model_output: torch.FloatTensor,
        step_index: int,
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[OpenSoraFlowMatchEulerSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            step_index (`float`):
                The current step index.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if not (
            isinstance(step_index, int)
            or isinstance(step_index, torch.IntTensor)
            or isinstance(step_index, torch.LongTensor)
        ): 
            raise ValueError("step_index should be an integer or a tensor of integer")

        if not (step_index >= 0 and step_index < len(self.sigmas)):
            raise ValueError("step_index should be in the range of [0, len(sigmas)]")
                             
        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)


        if not return_dict:
            return (prev_sample,)

        return OpenSoraFlowMatchEulerSchedulerOutput(prev_sample=prev_sample)
