# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2024 The HUAWEI Team. All rights reserved.

import os
from mindspeed_mm.data.data_utils.utils import TextProcesser
from mindspeed_mm.data.data_utils.mask_utils import STR_TO_TYPE

def is_video_file(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.3gp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in video_extensions

def is_image_file(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in image_extensions

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

    @staticmethod
    def conditional_pixel_values_checks(
        conditional_pixel_values_path,
        conditional_pixel_values_indices,
        mask_type,
    ):
        if conditional_pixel_values_path is None:
            raise ValueError("conditional_pixel_values_path should be provided")
        else:
            if not isinstance(conditional_pixel_values_path, list) or not isinstance(conditional_pixel_values_path[0], str):
                raise ValueError("conditional_pixel_values_path should be a list of strings")
    
        if not is_image_file(conditional_pixel_values_path[0]) and not is_video_file(conditional_pixel_values_path[0]):
            raise ValueError("conditional_pixel_values_path should be an image or video file path")  
        
        if is_video_file(conditional_pixel_values_path[0]) and len(conditional_pixel_values_path) > 1:
            raise ValueError("conditional_pixel_values_path should be a list of image file paths or a single video file path")
        
        if conditional_pixel_values_indices is not None \
            and (not isinstance(conditional_pixel_values_indices, list) or not isinstance(conditional_pixel_values_indices[0], int) \
                 or len(conditional_pixel_values_indices) != len(conditional_pixel_values_path)):
            raise ValueError("conditional_pixel_values_indices should be a list of integers with the same length as conditional_pixel_values_path")
        
        if mask_type is not None and not mask_type in STR_TO_TYPE.keys() and not mask_type in STR_TO_TYPE.values():
            raise ValueError(f"Invalid mask type: {mask_type}")

    def video_prompt_checks(self, video_prompt, kwargs):
        raise NotImplementedError()