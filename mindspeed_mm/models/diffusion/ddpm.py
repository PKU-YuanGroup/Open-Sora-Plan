# Modified from hojonathanho and OpenAI's diffusion repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DDPM:   https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
# IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# --------------------------------------------------------

from typing import List, Tuple, Dict, Callable, Union

import numpy as np
from tqdm.auto import tqdm
import torch
from torch import Tensor

from mindspeed_mm.utils.utils import get_device
from .diffusion_utils import (
    mean_flat,
    normal_kl,
    extract_into_tensor,
    discretized_gaussian_log_likelihood,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_beta_schedule,
    space_timesteps
)


class DDPM:
    """
    Contains utilities for the diffusion model.
    Arguments:
        model_mean_type: what the network predicts (x_{t-1}, x_0, or epsilon)
        model_var_type: which loss function (kl or unweighted MSE)
        loss_type: what is the variance of p(x_{t-1}|x_t) (learned, fixed to beta, or fixed to weighted beta)
                   what type of decoder, and how to weight its loss? is its variance learned too?
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
        **kwargs
    ):
        self.device = get_device(device)
        self.betas = get_beta_schedule(noise_schedule, num_train_steps).to(self.device)
        # init loss_type
        if use_kl:
            self.loss_type = LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            self.loss_type = LossType.RESCALED_MSE
        else:
            self.loss_type = LossType.MSE
        # init model_mean_type and model_var_type
        if predict_xstart:
            self.model_mean_type = ModelMeanType.START_X
        else:
            self.model_mean_type = ModelMeanType.EPSILON
        if learn_sigma:
            self.model_var_type = ModelVarType.LEARNED_RANGE
        elif not sigma_small:
            self.model_var_type = ModelVarType.FIXED_LARGE
        else:
            self.model_var_type = ModelVarType.FIXED_SMALL
        # space timesteps
        if num_inference_steps is not None and timestep_respacing is not None:
            timestep_respacing = str(num_inference_steps)
        elif timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [num_train_steps]
        use_timesteps = set(space_timesteps(num_train_steps, timestep_respacing))

        # init new_betas 
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        timestep_map = []
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        self.betas = torch.tensor(new_betas, device=self.device)
        self.num_timesteps = int(self.betas.shape[0])

        # Prepare alphas related constant
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0], device=self.device)])

        # Prepare constant coefficient, calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Prepare constant coefficient, calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = (
            torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
            if len(self.posterior_variance) > 1
            else torch.DoubleTensor([])
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_mean_variance(self, x_start: Tensor, t: Tensor) -> Tuple[Tensor]:
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor = None) -> Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError("the shape of noise and x_start must equal")
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)"""
        if x_start.shape != x_t.shape:
            raise AssertionError("the shape of x_start and x_t must equal")
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model_output: Tensor,
        x: Tensor,
        t: Tensor,
        clip_denoised: bool = True,
        denoised_fn: Callable = None,
    ) -> Dict:
        """
        Apply the model_output to predict the initial x, x_0.
        :param model_output: output of the PredictModel.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the x_start prediction before 
            it is used to sample. Applies before clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        B, C = x.shape[:2]
        if t.shape != (B,):
            raise AssertionError("the shape of t is wrong")
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            if model_output.shape != (B, C * 2, *x.shape[2:]):
                raise ValueError("the shape of t is wrong")
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract_into_tensor(torch.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    torch.cat(self.posterior_variance[1].unsqueeze(0), self.betas[1:]),
                    torch.log(torch.cat(self.posterior_variance[1].unsqueeze(0), self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x \
                          - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * model_output
            pred_xstart = process_xstart(pred_xstart)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        if not (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape):
            raise AssertionError("Shape does not match")
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def condition_mean(
        self,
        cond_fn: Callable,
        p_mean_var: Tensor,
        x: Tensor,
        t: Tensor,
        model_kwargs: dict = None
    ) -> Tensor:
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(
        self,
        cond_fn: Callable,
        p_mean_var: Tensor,
        x: Tensor,
        t: Tensor,
        model_kwargs: dict = None
    ) -> Tensor:
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        pred_xstart = p_mean_var["pred_xstart"]

        eps = (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - pred_xstart) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape
        )
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        if x.shape != eps.shape:
            raise AssertionError("Shape does not match")
        out["pred_xstart"] = (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * eps
        )
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x: Tensor,
        t: Tensor,
        clip_denoised: bool = True,
        denoised_fn: Callable = None,
        cond_fn: Callable = None,
        model_kwargs: Dict = None,
        mask: Tensor = None
    ) -> Dict:
        """
        Sample x_{t-1} from the model at the given timestep.
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
        if mask is not None:
            if mask.shape[0] != x.shape[0]:
                mask = mask.repeat(2, 1)  # HACK
            mask_t = (mask * len(self.betas)).to(torch.int)

            # x0: copy unchanged x values
            # x_noise: add noise to x values
            x0 = x.clone()
            x_noise = x0 * extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) + torch.randn_like(x
            ) * extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

            # active noise addition
            # WARNING: this is a hacky implementation
            mask_t_equall = (mask_t == t.unsqueeze(1))[:, None, :, None, None]
            x = torch.where(mask_t_equall, x_noise, x0)

            # create x_mask
            mask_t_upper = (mask_t > t.unsqueeze(1))[:, None, :, None, None]
            batch_size = x.shape[0]
            model_kwargs["x_mask"] = mask_t_upper.reshape(batch_size, -1).to(torch.bool)

        model_output = model(x, t, **model_kwargs)
        out = self.p_mean_variance(
            model_output,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        if mask is not None:
            mask_t_lower = (mask_t < t.unsqueeze(1))[:, None, :, None, None]
            sample = torch.where(mask_t_lower, x0, sample)

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_progressive(
        self,
        model,
        shape: Union[Tuple, List],
        noised_latents: Tensor = None,
        clip_denoised: bool = True,
        denoised_fn: Callable = None,
        cond_fn: Callable = None,
        model_kwargs: dict = None,
        progress: bool = False,
        mask: Tensor = None,
    ):
        """
        Generate samples from the model and yield intermediate samples from each timestep of diffusion.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noised_latents: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        Returns a generator over dicts, where each dict is the return value of p_sample().
        """
        if not isinstance(shape, (tuple, list)):
            raise AssertionError("param shape is incorrect")
        if noised_latents is None:
            noised_latents = torch.randn(*shape, device=self.device)
        
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            timestep = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    noised_latents,
                    timestep,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                )
                yield out
                latents = out["sample"]

    def sample(
        self,
        model,
        shape: Union[Tuple, List],
        noised_latents: Tensor = None,
        clip_denoised: bool = True,
        denoised_fn: Callable = None,
        cond_fn: Callable = None,
        model_kwargs: dict = None,
        progress: bool = False,
        mask: Tensor = None,
        **kwargs
    ) -> Tensor:
        """
        Generate samples from the model.
        Arguments are the same as p_sample_loop().
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noised_latents=noised_latents,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            progress=progress,
            mask=mask,
        ):
            final = sample
        return final["sample"]

    def _vb_terms_bpd(
        self,
        model_output: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        clip_denoised: bool = True,
        mask: Tensor = None
    ) -> Dict:
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model_output, x_t, t, clip_denoised=clip_denoised)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl, mask=mask) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll, mask=mask) / np.log(2.0)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        model_output: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        noise: Tensor = None,
        mask: Tensor = None,
        weights: Tensor = None,
        t: Tensor = None,
        **kwargs
    ) -> Tensor:
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            if mask is not None:
                raise AssertionError("mask not supported for KL loss")
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_output,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                if model_output.shape != (B, C * 2, *x_t.shape[2:]):
                    raise AssertionError("Shape does not match")
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model_output=frozen_out,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    mask=mask,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            if model_output.shape != target.shape == x_start.shape:
                raise AssertionError("Shape does not match")
            if weights is None:
                terms["mse"] = mean_flat((target - model_output) ** 2, mask=mask)
            else:
                weight = extract_into_tensor(weights, t, target.shape)
                terms["mse"] = mean_flat(weight * (target - model_output) ** 2, mask=mask)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
