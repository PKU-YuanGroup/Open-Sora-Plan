from typing import Optional, Union, List, Callable
import math
import inspect
import torch

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


class OpenSoraPlanPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model):
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                              predict_model=predict_model)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.transformer = predict_model

    @torch.no_grad()
    def __call__(self,
                 prompt,
                 prompt_embeds: Optional[torch.Tensor] = None,
                 negative_prompt: Optional[str] = None,
                 negative_prompt_embeds: Optional[torch.Tensor] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 eta: float = 0.0,
                 num_images_per_prompt: Optional[int] = 1,
                 num_frames: Optional[int] = None,
                 guidance_scale: float = 4.5,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 max_length: Optional[int] = 300,
                 clean_caption: bool = True,
                 mask_feature: bool = True,
                 enable_temporal_attentions: bool = True,
                 added_cond_kwargs: dict = None,
                 ):

        # 1. Check inputs.
        # text prompt checks
        self.text_prompt_checks(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)
        self.generate_params_checks(height, width)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.text_encoder.device or self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_texts(
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            max_length=max_length,
            clean_caption=True)
        # 3.1 reshape prompt embeddings
        prompt_embeds = self.reshape_prompt_embeddings(prompt_embeds, num_images_per_prompt)
        # 3.2 reshape prompt mask
        prompt_embeds_attention_mask = self.reshape_prompt_mask(prompt_embeds_attention_mask, batch_size,
                                                                num_images_per_prompt)
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt_embeds = self.reshape_prompt_embeddings(negative_prompt_embeds, num_images_per_prompt)

        if mask_feature:
            prompt_embeds = prompt_embeds.unsqueeze(1)
            masked_prompt_embeds, keep_indices = self.mask_text_embeddings(prompt_embeds, prompt_embeds_attention_mask)
            masked_prompt_embeds = masked_prompt_embeds.squeeze(1)
            masked_negative_prompt_embeds = (
                negative_prompt_embeds[:, :keep_indices, :] if negative_prompt_embeds is not None else None
            )
            prompt_embeds = masked_prompt_embeds
            negative_prompt_embeds = masked_negative_prompt_embeds

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 5. Prepare latents
        latent_channels = self.transformer.config.in_channels
        batch_size = batch_size * num_images_per_prompt
        shape = (
            batch_size,
            latent_channels,
            (math.ceil((int(num_frames) - 1) / self.vae.vae_scale_factor[0]) + 1) if int(
                num_frames) % 2 == 1 else math.ceil(int(num_frames) / self.vae.vae_scale_factor[0]),
            math.ceil(int(height) / self.vae.vae_scale_factor[1]),
            math.ceil(int(width) / self.vae.vae_scale_factor[2]),
        )
        latents = self.prepare_latents(shape, generator=generator, device=device, dtype=prompt_embeds.dtype,
                                       latents=latents)
        # 6 prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        # opensoraplan args
        model_kwargs = {"encoder_hidden_states": prompt_embeds,
                        "added_cond_kwargs": added_cond_kwargs,
                        "enable_temporal_attentions": enable_temporal_attentions,
                        "return_dict": False}

        latents = self.scheduler.sample(model=self.transformer, shape=shape, latents=latents, model_kwargs=model_kwargs,
                                        extra_step_kwargs=extra_step_kwargs)
        video = self.decode_latents(latents.to(self.vae.vae.dtype))  # TODO vae是个warpper里面包裹一层vae 使用mindspeed套件模型后去掉
        video = video.permute(0, 2, 1, 3, 4)  # [b,t,c,h,w -> [b,c,t,h,w]

        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
