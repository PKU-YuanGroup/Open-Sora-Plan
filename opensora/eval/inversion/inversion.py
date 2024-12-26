
from typing import List, Union, Optional, Dict
from enum import Enum
from tqdm import tqdm
import bisect

import torch
try:
    import torch_npu
    from npu_config import npu_config
except:
    torch_npu = None
    npu_config = None


@torch.inference_mode()
def flow_matching_inversion(
    model,
    scheduler,
    latents,
    attention_mask,
    encoder_hidden_states,
    encoder_attention_mask,
    encoder_hidden_states_2,
    encoder_attention_mask_2,
    pooled_projections,
    sigmas=None,
    do_classifier_free_guidance=True,
    guidance_scale=7.0,
    num_inference_steps=100,
    num_inverse_steps=80,
    resample=False,
    inverse_cache_dict: Optional[Dict] = {},
    **kwargs,
):
    orig_sigmas = sigmas      
    sigmas = scheduler.set_sigmas(num_inference_steps, device=latents.device, sigmas=sigmas, inversion=True, **kwargs)
    
    # Get cache
    cached_steps = sorted(list(inverse_cache_dict.keys()))
    last_cached_step_idx = bisect.bisect_left(cached_steps, num_inverse_steps) - 1
    if last_cached_step_idx < 0:
        start_step = 0
    else:
        start_step = cached_steps[last_cached_step_idx]
        latents = inverse_cache_dict[start_step]
        print(f"Start from cached step - {start_step}.")
    
    for step in tqdm(range(start_step, num_inverse_steps), desc="Inversing"):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        step_index = step
        timestep = (sigmas[step_index] * 1000).expand(latent_model_input.shape[0])
        noise_pred = model(
            latent_model_input,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_dict=False,
            **kwargs,
        )[0]
        if torch.any(torch.isnan(noise_pred)):
            raise ValueError("NaN in noise_pred")
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, step_index, latents, return_dict=False)[0]

    inverse_cache_dict[num_inverse_steps] = latents
    
    if resample:
        sigmas = scheduler.set_sigmas(num_inference_steps, device=latents.device, sigmas=orig_sigmas, inversion=False, **kwargs)
        if kwargs.get("encoder_hidden_states_target", None) is not None:
            encoder_hidden_states = kwargs.get("encoder_hidden_states_target")
        if kwargs.get("encoder_attention_mask_target", None) is not None:
            encoder_attention_mask = kwargs.get("encoder_attention_mask_target")
        if kwargs.get("encoder_hidden_states_2_target", None) is not None:
            encoder_hidden_states_2 = kwargs.get("encoder_hidden_states_2_target")
        if kwargs.get("encoder_attention_mask_2_target", None) is not None:
            encoder_attention_mask_2 = kwargs.get("encoder_attention_mask_2_target")
        if kwargs.get("pooled_projections_target", None) is not None:
            pooled_projections = kwargs.get("pooled_projections_target")
        for step in tqdm(range(num_inverse_steps), desc="Resampling"):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            step_index = num_inference_steps - num_inverse_steps + step
            timestep = (sigmas[step_index] * 1000).expand(latent_model_input.shape[0])
            noise_pred = model(
                latent_model_input,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states_2=encoder_hidden_states_2,
                encoder_attention_mask_2=encoder_attention_mask_2,
                timestep=timestep,
                pooled_projections=pooled_projections,
                return_dict=False,
                **kwargs,
            )[0]
            if torch.any(torch.isnan(noise_pred)):
                raise ValueError("NaN in noise_pred")

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, step_index, latents, return_dict=False)[0]

    return latents
