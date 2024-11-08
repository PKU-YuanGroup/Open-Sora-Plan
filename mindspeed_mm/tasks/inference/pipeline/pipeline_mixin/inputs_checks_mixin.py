# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2024 The HUAWEI Team. All rights reserved.


from mindspeed_mm.data.data_utils.utils import TextProcesser


class InputsCheckMixin:

    @staticmethod
    def text_prompt_checks(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds):

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @staticmethod
    def generate_params_checks(height, width):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    @staticmethod
    def _preprocess_text(prompt, clean, to_lower):
        if clean:
            prompt = TextProcesser.text_preprocessing(prompt)
        else:
            if to_lower:
                prompt = prompt.lower().strip()
            else:
                prompt = prompt.strip()
        return prompt

    @staticmethod
    def preprocess_text(prompt, clean, to_lower=True):
        if not isinstance(prompt, (tuple, list)):
            prompt = [prompt]
        return [InputsCheckMixin._preprocess_text(prompt, clean, to_lower) for prompt in prompt]

    def image_prompt_checks(self, image_prompt, ):
        raise NotImplementedError()

    def video_prompt_checks(self, video_prompt, kwargs):
        raise NotImplementedError()