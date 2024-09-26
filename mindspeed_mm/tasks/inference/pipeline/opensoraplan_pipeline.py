from typing import Optional, Union, List, Callable
import math
import inspect
import torch

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin
from mindspeed_mm.tasks.inference.pipeline.patchs.sora_patchs import replace_with_fp32_forwards


class OpenSoraPlanPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model):
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                              predict_model=predict_model)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.predict_model = predict_model
        replace_with_fp32_forwards()

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
                 max_sequence_length: Optional[int] = 300,
                 clean_caption: bool = True,
                 mask_feature: bool = True,
                 enable_temporal_attentions: bool = True,
                 added_cond_kwargs: dict = None,
                 use_prompt_template: bool = True,
                 motion_score: float = 1.0,
                 **kwargs,
                 ):

        # 1. Check inputs.
        # text prompt checks
        if use_prompt_template:
            prompt, negative_prompt = self.use_prompt_template(positive_prompt=prompt, negative_prompt=negative_prompt)
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
            negative_prompt=negative_prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            max_length=max_sequence_length,
            clean_caption=clean_caption)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_embeds_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_embeds_attention_mask],
                                                     dim=0)

        # 5. Prepare latents
        latent_channels = self.predict_model.in_channels
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

        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
        if prompt_embeds_attention_mask.ndim == 2:
            prompt_embeds_attention_mask = prompt_embeds_attention_mask.unsqueeze(1)  # b l -> b 1 l
        model_kwargs = {"encoder_hidden_states": prompt_embeds,
                        "added_cond_kwargs": added_cond_kwargs,
                        "enable_temporal_attentions": enable_temporal_attentions,
                        "encoder_attention_mask": prompt_embeds_attention_mask,
                        "motion_score": motion_score,
                        "return_dict": False}

        latents = self.scheduler.sample(model=self.predict_model, shape=shape, latents=latents, model_kwargs=model_kwargs,
                                        extra_step_kwargs=extra_step_kwargs)
        video = self.decode_latents(latents.to(self.vae.dtype))

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

    @staticmethod
    def use_prompt_template(positive_prompt, negative_prompt):
        positive_template_list = []
        negative_template_list = []
        if not negative_prompt:
            negative_prompt = ""
        positive_template = "(masterpiece), (best quality), (ultra-detailed), {}. emotional, harmonious, vignette, " \
                            "4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
        negative_template = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, " \
                            "fewer digits, cropped, worst quality, low quality, normal quality, " \
                            "jpeg artifacts, signature, watermark, username, blurry"
        if isinstance(positive_prompt, (list, tuple)):
            for positive_prompt_i in positive_prompt:
                positive_template_i = positive_template.format(positive_prompt_i)
                negative_template_i = negative_template + negative_prompt
                positive_template_list.append(positive_template_i)
                negative_template_list.append(negative_template_i)
            return positive_template_list, negative_template_list
        else:
            positive_template_i = positive_template.format(positive_prompt)
            negative_template_i = negative_template + negative_prompt
            return [positive_template_i], [negative_template_i]
