from typing import Optional, Union, List, Callable

import torch

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


class OpenSoraPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                              predict_model=predict_model)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.predict_model = predict_model
        text_encoder.use_attention_mask = config.use_attention_mask
        self.num_frames, self.height, self.width = config.input_size
        if config.use_y_embedder:
            text_encoder.y_embedder = predict_model.y_embedder

    @torch.no_grad()
    def __call__(self,
                 prompt,
                 prompt_embeds: Optional[torch.Tensor] = None,
                 negative_prompt: Optional[str] = None,
                 negative_prompt_embeds: Optional[torch.Tensor] = None,
                 num_images_per_prompt: Optional[int] = 1,
                 latents: Optional[torch.FloatTensor] = None,
                 clean_caption: bool = True,
                 fps: int = None,
                 model_args: Optional[dict] = None,
                 device: torch.device = "npu",
                 dtype: torch.dtype = None,
                 **kwargs
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
        model_args["fps"] = torch.tensor([fps], device=device, dtype=dtype).repeat(batch_size)
        model_args["height"] = torch.tensor([self.height], device=device, dtype=dtype).repeat(batch_size)
        model_args["width"] = torch.tensor([self.width], device=device, dtype=dtype).repeat(batch_size)
        model_args["num_frames"] = torch.tensor([self.num_frames], device=device, dtype=dtype).repeat(batch_size)
        model_args["ar"] = torch.tensor([self.height / self.width], device=device, dtype=dtype).repeat(batch_size)
        model_args["mask"] = prompt_embeds_attention_mask

        # 5. Prepare latents
        image_size = (self.height, self.width)
        batch_size = batch_size * num_images_per_prompt
        input_size = (self.num_frames, *image_size)
        latent_size = self.vae.get_latent_size(input_size)
        shape = (batch_size, self.vae.out_channels, *latent_size)
        z = torch.randn(shape, device=device, dtype=dtype)
        masks = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
        latents = self.scheduler.sample(model=self.predict_model, shape=shape, clip_denoised=False, latents=z,
                                        mask=masks,
                                        model_kwargs=model_args, progress=True)  # b,c,t,h,w
        video = self.decode_latents(latents.to(self.vae.dtype), num_frames=self.num_frames)  # b,c,t,h,w
        video = video[:, :self.num_frames, :self.height, :self.width]

        return video

    def null(self, n):
        null_y = self.text_encoder.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y
