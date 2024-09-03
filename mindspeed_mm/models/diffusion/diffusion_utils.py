# Modified from OpenAI's diffusion repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
# IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# --------------------------------------------------------

import enum
from typing import List, Union, Set

import numpy as np
import torch
from torch import Tensor
from einops import rearrange


def get_beta_schedule(schedule_name: str, num_diffusion_timesteps: torch.int) -> Tensor:
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64)
    elif schedule_name == "squaredcos_cap_v2":
        max_beta = 0.999
        betas = []
        
        def alpha_bar(t):
            return torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2

        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.DoubleTensor(betas)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def normal_kl(mean1: Tensor, logvar1: Tensor, mean2: Tensor, logvar2: Tensor) -> Tensor:
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    return 0.5 * (
        -1.0 + logvar2 - logvar1 \
        + torch.exp(logvar1 - logvar2) \
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x) -> Tensor:
    """
    A fast approximation of the cumulative distribution function of thestandard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))


def continuous_gaussian_log_likelihood(x: Tensor, means: Tensor, log_scales: Tensor) -> Tensor:
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(normalized_x)
    return log_probs


def discretized_gaussian_log_likelihood(x: Tensor, means: Tensor, log_scales: Tensor) -> Tensor:
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    if not x.shape == means.shape == log_scales.shape:
        raise AssertionError("Shape does not match")
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs


def space_timesteps(num_timesteps: Tensor, section_counts: Union[str, List[int]]) -> Set:
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def extract_into_tensor(arr: Tensor, timesteps: Tensor, broadcast_shape: List[int]) -> Tensor:
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    """
    Which type of loss the model use.
    MSE: use raw MSE loss (and KL when learning variances)
    RESCALED_MSE: use raw MSE loss (with RESCALED_KL when learning variances)
    KL: use the variational lower-bound
    RESCALED_KL: like KL, but rescale to estimate the full VLB
    """
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def mean_flat(tensor: Tensor, mask: Tensor = None) -> Tensor:
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        if tensor.dim() != 5:
            raise AssertionError("tensor shape must be 5")
        if tensor.shape[2] != mask.shape[1]:
            raise AssertionError("tensor.shape[2] must equal to mask.shape[1]")
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss
