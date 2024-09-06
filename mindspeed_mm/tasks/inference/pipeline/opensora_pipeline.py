from typing import Optional, Union, List, Callable

import torch

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


class OpenSoraPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):

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
                 clean_caption: bool = True,
                 mask_feature: bool = True,
                 enable_temporal_attentions: bool = True,
                 added_cond_kwargs: dict = None,
                 model_args: Optional[dict] = None,
                 device: torch.device = "npu",
                 dtype: torch.dtype = None,
                 ):

        # 1 check prompts
        self.text_prompt_checks(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)
        prompt = self.preprocess_text(prompt, clean=clean_caption)

        # 2
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_attention_mask, _, _ = self.encode_texts(prompt=prompt, device=device,
                                                                              do_classifier_free_guidance=False)
        prompt_embeds = prompt_embeds[:, None]
        if model_args:
            model_args.update(dict(prompt=prompt_embeds, prompt_mask=prompt_embeds_attention_mask))
        else:
            model_args = dict(prompt=prompt_embeds, prompt_mask=prompt_embeds_attention_mask)
        y_null = self.null(batch_size)
        model_args["prompt"] = torch.cat([model_args["prompt"], y_null], 0)

        # 5. Prepare latents
        image_size = (height, width)
        batch_size = batch_size * num_images_per_prompt
        input_size = (num_frames, *image_size)
        latent_size = self.vae.get_latent_size(input_size)
        shape = (batch_size, self.vae.out_channels, *latent_size)
        z = torch.randn(shape, device=device, dtype=dtype)
        z = torch.cat([z, z], 0)
        shape = z.shape

        latents = self.scheduler.sample(model=self.transformer, shape=shape, clip_denoised=False, latents=z,
                                        model_kwargs=model_args, progress=True)  # b,c,t,h,w
        latents, _ = latents.chunk(2, dim=0)
        video = self.decode_latents(latents.to(self.vae.dtype))  # b,c,t,h,w
        video = video[:, :num_frames, :height, :width]

        return video

    def null(self, n):
        null_y = self.text_encoder.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y
