from typing import Optional, Union, List, Callable
import math
import inspect

import torch

from mindspeed_mm.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin
from mindspeed_mm.inference.pipeline.patchs.sora_patchs import replace_with_fp32_forwards
from mindspeed_mm.utils.mask_utils import MaskProcessor, MaskCompressor, TYPE_TO_STR
from mindspeed_mm.inference.pipeline.utils.sora_utils import get_pixel_values, get_mask_type_cond_indices, get_video_transform, get_resize_transform


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
        self.model_type = config.model_type
        if self.model_type == "i2v":
            self.mask_processor = MaskProcessor(min_clear_ratio=0.5, max_clear_ratio=0.5)
            self.mask_compressor = MaskCompressor(ae_stride_t=self.vae.vae_scale_factor[0],
                                                  ae_stride_h=self.vae.vae_scale_factor[1],
                                                  ae_stride_w=self.vae.vae_scale_factor[2])
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
        if self.model_type == "i2v":
            conditional_pixel_values_path = kwargs.get("conditional_pixel_values_path", None)
            mask_type = kwargs.get("mask_type", None)
            conditional_pixel_values_indices = kwargs.get("conditional_pixel_values_indices", None)
            crop_for_hw = kwargs.get("crop_for_hw", False)
            max_hxw = kwargs.get("max_hxw", 236544)
            pixel_values = []
            pixel_values_indices = []
            for pixel_values_path_i in conditional_pixel_values_path:
                self.i2v_prompt_checks(pixel_values_path_i, mask_type)
                mask_type, pixel_values_indices_i = get_mask_type_cond_indices(mask_type,
                                                                                    pixel_values_path_i,
                                                                                    conditional_pixel_values_indices,
                                                                                    self.num_frames)

                pixel_values_i = get_pixel_values(pixel_values_path_i, self.num_frames)
                pixel_values += pixel_values_i
                pixel_values_indices.append(pixel_values_indices_i)
            min_height = min([pixels.shape[2] for pixels in pixel_values])
            min_width = min([pixels.shape[3] for pixels in pixel_values])

            resize_transform = get_resize_transform(
                ori_height=min_height,
                ori_width=min_width,
                height=self.height,
                width=self.width,
                crop_for_hw=crop_for_hw,
                max_hxw=max_hxw,
            )

            video_transform = get_video_transform()
            pixel_values = torch.cat([resize_transform(pixels) for pixels in pixel_values])
            height, width = pixel_values.shape[-2], pixel_values.shape[-1]

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
        if self.model_type == "i2v":
            masked_pixel_values, mask = self.get_masked_pixel_values_mask(
                pixel_values,
                pixel_values_indices,
                mask_type,
                batch_size,
                num_images_per_prompt,
                self.num_frames,
                height,
                width,
                video_transform,
                prompt_embeds.dtype,
                device
            )

            i2v_kwargs = {MASKED_VIDEO: masked_pixel_values, INPUT_MASK: mask}

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
        if self.model_type == "i2v":
            model_kwargs.update(i2v_kwargs)

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

            negative_template = """
            Worst quality, Normal quality, Low quality, Low res, Blurry, Jpeg artifacts, Grainy, watermark, banner, 
            Cropped, Out of frame, Out of focus, Bad anatomy, Bad proportions, Deformed, Disconnected limbs, Disfigured, 
            username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, overexposed, underexposed.
            """
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

    def get_masked_pixel_values_mask(
        self,
        conditional_pixel_values,
        conditional_pixel_values_indices,
        mask_type,
        batch_size,
        num_samples_per_prompt,
        num_frames,
        height,
        width,
        video_transform,
        weight_dtype,
        device
    ):
        if device is None:
            device = getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')

        conditional_pixel_values = conditional_pixel_values.to(device=device, dtype=weight_dtype)

        masked_pixel_values = []
        mask = []
        for i in range(batch_size):
            conditional_pixel_values_i = conditional_pixel_values[i].unsqueeze(0)
            conditional_pixel_values_indices_i = conditional_pixel_values_indices[i]
            if conditional_pixel_values_i.shape[0] == num_frames:
                inpaint_cond_data = self.mask_processor(conditional_pixel_values_i, mask_type=mask_type)
                masked_pixel_values_i, mask_i = inpaint_cond_data['masked_pixel_values'], inpaint_cond_data['mask']
            else:
                input_pixel_values = torch.zeros([num_frames, 3, height, width], device=device, dtype=weight_dtype)
                input_mask = torch.ones([num_frames, 1, height, width], device=device, dtype=weight_dtype)
                input_pixel_values[conditional_pixel_values_indices_i] = conditional_pixel_values_i
                input_mask[conditional_pixel_values_indices_i] = 0
                masked_pixel_values_i = input_pixel_values * (input_mask < 0.5)
                mask_i = input_mask

            print('conditional_pixel_values_indices_i', conditional_pixel_values_indices_i)
            print('mask_type', TYPE_TO_STR[mask_type])

            masked_pixel_values_i = video_transform(masked_pixel_values_i)

            masked_pixel_values_i = masked_pixel_values_i.unsqueeze(0).repeat(num_samples_per_prompt, 1, 1, 1,
                                                                          1).transpose(1, 2).contiguous()  # 1 c t h w
            mask_i = mask_i.unsqueeze(0).repeat(num_samples_per_prompt, 1, 1, 1, 1).transpose(1, 2).contiguous()  # 1 c t h w

            masked_pixel_values.append(masked_pixel_values_i)
            mask.append(mask_i)

        masked_pixel_values = torch.cat(masked_pixel_values, dim=0)
        mask = torch.cat(mask, dim=0)
        masked_pixel_values = masked_pixel_values.to(self.vae.dtype)
        masked_pixel_values = self.vae.encode(masked_pixel_values)

        mask = self.mask_compressor(mask)

        masked_pixel_values = torch.cat(
            [masked_pixel_values] * 2) if self.do_classifier_free_guidance else masked_pixel_values
        mask = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask

        masked_pixel_values = masked_pixel_values.to(weight_dtype)
        mask = mask.to(weight_dtype)

        return masked_pixel_values, mask
