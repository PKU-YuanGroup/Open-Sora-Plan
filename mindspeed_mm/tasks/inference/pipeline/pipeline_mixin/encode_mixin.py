# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2024 The HUAWEI Team. All rights reserved.
from typing import List, Optional
import logging as logger

import torch
import transformers

from mindspeed_mm.tasks.inference.pipeline.utils.llava_utils import expand2square, process_anyres_image
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


class MMEncoderMixin:

    def encode_texts(self,
                     prompt,
                     device,
                     do_classifier_free_guidance=False,
                     negative_prompt=None,
                     prompt_embeds: Optional[torch.FloatTensor] = None,
                     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                     clip_skip: Optional[int] = None,
                     clean_caption=True,
                     max_length: Optional[int] = None,
                     prompt_to_lower=True
                     ):
        max_length = max_length if max_length else self.tokenizer.model_max_length

        if device is None:
            device = self.text_encoder.device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, InputsCheckMixin):
                prompt = self.preprocess_text(prompt, clean_caption, prompt_to_lower)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )
            if hasattr(self.text_encoder,
                       "use_attention_mask") and self.text_encoder.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            prompt_embeds_attention_mask = attention_mask
            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                if isinstance(prompt_embeds, transformers.utils.ModelOutput):
                    prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)
        else:
            if hasattr(self.text_encoder, "use_attention_mask") and self.text_encoder.use_attention_mask:
                prompt_embeds_attention_mask = torch.ones_like(prompt_embeds)
            else:
                prompt_embeds_attention_mask = None

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, InputsCheckMixin):
                uncond_tokens = self.preprocess_text(uncond_tokens, clean_caption)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            #  参数控制
            if hasattr(self.text_encoder,
                       "use_attention_mask") and self.text_encoder.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            negative_prompt_attention_mask = attention_mask
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            if isinstance(negative_prompt_embeds, transformers.utils.ModelOutput):
                negative_prompt_embeds = negative_prompt_embeds[0]
        else:
            if hasattr(self.text_encoder,
                       "use_attention_mask") and self.text_encoder.use_attention_mask and negative_prompt_embeds is not None:
                negative_prompt_attention_mask = torch.ones_like(negative_prompt_embeds)
            else:
                negative_prompt_attention_mask = None

        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_embeds_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @staticmethod
    def mask_text_embeddings(emb, mask):
        if emb.shape[0] == 1:
            keep_index = mask.sum().item()
            return emb[:, :, :keep_index, :], keep_index  # 1, 120, 4096 -> 1 7 4096
        else:
            masked_feature = emb * mask[:, None, :, None]  # 1 120 4096
            return masked_feature, emb.shape[2]

    @staticmethod
    def reshape_prompt_embeddings(prompt_embeds, num_images_per_prompt):
        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    @staticmethod
    def reshape_prompt_mask(prompt_mask, batch_size, num_images_per_prompt):
        prompt_attention_mask = prompt_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        return prompt_attention_mask

    def process_images(self, images: List, image_aspect_ratio: float = None, image_grid_pinpoints: Optional = None):
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
        elif image_aspect_ratio == "anyres":
            for image in images:
                image = process_anyres_image(image, self.image_processor, image_grid_pinpoints)
                new_images.append(image)
        else:
            return self.image_processor(images, return_tensors='pt')['pixel_values']
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    def encode_videos(self, video_prompt, video_encoder, kwargs):
        pass
