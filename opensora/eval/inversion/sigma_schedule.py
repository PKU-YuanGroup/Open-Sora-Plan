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

# NOTE Implementation of Mochi 
def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule

def opensora_linear_quadratic_schedule(num_inference_steps, approximate_steps=1000):
    assert approximate_steps % 2 == 0, "approximate_steps must be even"
    assert num_inference_steps % 2 == 0, "num_inference_steps must be even"
    assert num_inference_steps <= approximate_steps, "num_inference_steps must be less than or equal to approximate_steps"

    _num_inference_steps = num_inference_steps // 2
    _approximate_steps = approximate_steps // 2

    linear_sigmas = [i / (2 * _approximate_steps) for i in range(_num_inference_steps)]
    # NOTE we define a quadratic schedule that is f(x) = ax^2 + bx + c
    quadratic_a = (_approximate_steps - _num_inference_steps) / (_approximate_steps * _num_inference_steps ** 2)
    quadratic_b = (5 * _num_inference_steps - 4 * _approximate_steps) / (2 * _approximate_steps * _num_inference_steps)
    quadratic_c = (_approximate_steps - _num_inference_steps) / _approximate_steps
    quadratic_sigmas = [
        quadratic_a * i ** 2 + quadratic_b * i + quadratic_c for i in range(_num_inference_steps, 2 * _num_inference_steps)
    ]
    sigmas = linear_sigmas + quadratic_sigmas + [1.0]
    sigmas = [1.0 - x for x in sigmas]
    return sigmas

