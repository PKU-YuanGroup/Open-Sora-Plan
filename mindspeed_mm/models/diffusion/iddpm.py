# Modified from OpenSoraPlan and OpenAI's diffusion repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenSoraPlan: https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/v1.1.0/opensora/models/diffusion/diffusion
# IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# --------------------------------------------------------

from typing import List, Union, Tuple, Callable, Dict

import torch
from torch import Tensor
from tqdm.auto import tqdm

from .diffusion_utils import extract_into_tensor
from .ddpm import DDPM


class IDDPM(DDPM):
    """
    Improved DDPM for diffusion model.
    """

    def __init__(
        self,
        num_inference_steps: int = None,
        num_train_steps: int = 1000,
        timestep_respacing: Union[str, List] = None,
        noise_schedule: str = "linear",
        use_kl: bool = False,
        sigma_small: bool = False,
        predict_xstart: bool = False,
        learn_sigma: bool = True,
        rescale_learned_sigmas: bool = False,
        device: str = "npu",
        **kwargs,
    ):
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_steps=num_train_steps,
            timestep_respacing=timestep_respacing,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            sigma_small=sigma_small,
            predict_xstart=predict_xstart,
            learn_sigma=learn_sigma,
            rescale_learned_sigmas=rescale_learned_sigmas,
            device=device,
            **kwargs,
        )
        self.scale = kwargs.get("scale", None)
        self.channel = kwargs.get("channel", None)

    def ddim_sample(
        self,
        model,
        x: Tensor,
        t: Tensor,
        clip_denoised: bool = True,
        denoised_fn: Callable = None,
        cond_fn: Callable = None,
        model_kwargs: Dict = None,
        eta: float = 0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        new_ts = self.map_tensors[t].to(device=t.device, dtype=t.dtype)

        half = x[: len(x) // 2]
        x = torch.cat([half, half], dim=0)
        model_output = model(x, new_ts, **model_kwargs)

        model_output = model_output["x"] if isinstance(model_output, dict) else model_output
        if self.scale is None:
            raise Exception("scale cannot be None")
        if self.channel is None:
            self.channel = model_output.shape[1] // 2
        eps, rest = model_output[:, :self.channel], model_output[:, self.channel:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        model_output = torch.cat([eps, rest], dim=1)


        out = self.p_mean_variance(
            model_output,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        pred_xstart = out["pred_xstart"]
        eps = (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop_progressive(
        self,
        model: callable,
        shape: Union[Tuple, List],
        latents: Tensor = None,
        clip_denoised: bool = True,
        denoised_fn: Callable = None,
        cond_fn: Callable = None,
        model_kwargs: Dict = None,
        progress: bool = False,
        eta: float = 0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if not isinstance(shape, (tuple, list)):
            raise AssertionError("param shape must be tuple or list")
        if latents is None:
            latents = torch.randn(*shape, device=self.device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            indices = tqdm(indices)
        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    latents,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                latents = out["sample"]

    def sample(
        self,
        model: callable,
        shape: Union[Tuple, List],
        latents: Tensor = None,
        clip_denoised: bool = True,
        denoised_fn: Callable = None,
        cond_fn: Callable = None,
        model_kwargs: Dict = None,
        progress: bool = False,
        eta: float = 0.0,
        **kwargs,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        latents = torch.cat([latents, latents], 0)
        shape = latents.shape
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            latents=latents,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            progress=progress,
            eta=eta,
        ):
            final = sample
        sample, _ = final["sample"].chunk(2, dim=0)
        return sample
