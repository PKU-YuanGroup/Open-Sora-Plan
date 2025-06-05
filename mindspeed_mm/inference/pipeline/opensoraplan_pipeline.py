from typing import Optional, Union, List, Callable
import math
import inspect

import torch

from mindspeed_mm.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin
from mindspeed_mm.inference.pipeline.patchs.sora_patchs import replace_with_fp32_forwards

class OpenSoraPlanPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):

    def __init__(
        self, 
        vae,
        text_encoder, 
        tokenizer, 
        scheduler, 
        predict_model,
        text_encoder_2=None,
        tokenizer_2=None,
        config=None
    ):
        self.register_modules(
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
            vae=vae, 
            scheduler=scheduler,
            predict_model=predict_model
        )

        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.predict_model = predict_model
        text_encoder.use_attention_mask = config.use_attention_mask
        self.num_frames, self.height, self.width = config.input_size
        self.version = config.version
        replace_with_fp32_forwards()

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        num_samples_per_prompt: Optional[int] = 1,
        guidance_scale: float = 4.5,
        guidance_rescale: float = 0.7,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_sequence_length: Optional[int] = 300,
        clean_caption: bool = True,
        added_cond_kwargs: dict = None,
        use_prompt_template: bool = True,
        use_prompt_preprocess: bool = True,
        **kwargs,
    ):

        # 1. Check inputs.
        # text prompt checks
        if use_prompt_template:
            prompt, negative_prompt = self.prompt_template(positive_prompt=prompt, negative_prompt=negative_prompt)

        print("prompt: ", prompt, "negative_prompt: ", negative_prompt)
        self.text_prompt_checks(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)
        self.generate_params_checks(self.height, self.width)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.text_encoder.device or self._execution_device

        self.do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_texts(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            num_samples_per_prompt=num_samples_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            max_length=512,
            clean_caption=clean_caption,
            use_prompt_preprocess=use_prompt_preprocess
        )
        # print(f"prompt_embeds: {prompt_embeds.shape}, negative_prompt_embeds: {negative_prompt_embeds.shape}, prompt_embeds_attention_mask: {prompt_embeds_attention_mask.shape}, negative_prompt_attention_mask: {negative_prompt_attention_mask.shape}")
        if self.tokenizer_2 is not None:
            prompt_embeds_2, prompt_embeds_attention_mask_2, negative_prompt_embeds_2, negative_prompt_attention_mask_2 = self.encode_texts(
                tokenizer=self.tokenizer_2,
                text_encoder=self.text_encoder_2,
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=device,
                num_samples_per_prompt=num_samples_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                max_length=77,
                clean_caption=clean_caption,
                use_prompt_preprocess=use_prompt_preprocess
            )
        else:
            prompt_embeds_2, prompt_embeds_attention_mask_2, negative_prompt_embeds_2, negative_prompt_attention_mask_2 = None, None, None, None


        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_embeds_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_embeds_attention_mask], dim=0)
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2], dim=0)
        # 5. Prepare latents
        latent_channels = self.predict_model.in_channels
        batch_size = batch_size * num_samples_per_prompt
        shape = (
            batch_size,
            latent_channels,
            (math.ceil((int(self.num_frames) - 1) / self.vae.vae_scale_factor[0]) + 1) if int(
                self.num_frames) % 2 == 1 else math.ceil(int(self.num_frames) / self.vae.vae_scale_factor[0]),
            math.ceil(int(self.height) / self.vae.vae_scale_factor[1]),
            math.ceil(int(self.width) / self.vae.vae_scale_factor[2]),
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
        if prompt_embeds_2 is not None and prompt_embeds_2.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d
        model_kwargs = {"prompt_embeds": prompt_embeds,
                        "prompt_embeds_2": prompt_embeds_2,
                        "added_cond_kwargs": added_cond_kwargs,
                        "prompt_attention_mask": prompt_embeds_attention_mask,
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

    def prompt_template(self, positive_prompt, negative_prompt):
        positive_template_list = []
        negative_template_list = []
        if not negative_prompt:
            negative_prompt = ""
        if self.version == "v1.2":
            positive_template = "(masterpiece), (best quality), (ultra-detailed), {}. emotional, harmonious, vignette, " \
                                "4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
            negative_template = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, " \
                                "fewer digits, cropped, worst quality, low quality, normal quality, " \
                                "jpeg artifacts, signature, watermark, username, blurry"
        elif self.version == "v1.3":
            positive_template = """
            high quality, high aesthetic, {}
            """
            negative_template = """
            nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
            low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
            """
        elif self.version == "v1.5":
            positive_template = """
            high quality, {}
            """

            negative_template = "Worst quality, Normal quality, Low quality, Low res, Blurry details, Jpeg artifacts, Grainy, watermark, too garish, " \
            "Cropped, Out of frame, Out of focus, Transition, Jump cut, Bad anatomy, Bad proportions, Deformed, Disconnected limbs, Disfigured, " \
            "duplicate, ugly, monochrome, mutation, disgusting, overexposed, underexposed, Subtitles, Motionless, Overall gray, Extra fingers, " \
            "Poorly drawn hands, Poorly drawn face, Fused fingers, Extra legs, Walking backward."
        else:
            positive_template = "{}"
            negative_template = ""
            
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
