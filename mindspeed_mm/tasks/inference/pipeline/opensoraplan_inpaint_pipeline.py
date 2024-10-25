from typing import Optional, Union, List, Callable

import os
import math
import inspect
import decord
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda, Resize
from einops import rearrange


from mindspeed_mm.data.data_utils.data_transform import CenterCropResizeVideo, SpatialStrideCropVideo,ToTensorAfterResize, maxhwresize
from mindspeed_mm.data.data_utils.mask_utils import MaskProcessor, MaskCompressor, MaskType, STR_TO_TYPE, TYPE_TO_STR, GaussianNoiseAdder
from mindspeed_mm.tasks.inference.pipeline.opensoraplan_pipeline import OpenSoraPlanPipeline

def is_video_file(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.3gp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in video_extensions

def is_image_file(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in image_extensions

def open_image(file_path):
    image = Image.open(file_path).convert("RGB")
    return image

def open_video(file_path, start_frame_idx, num_frames, frame_interval=1):

    decord_vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=1)

    total_frames = len(decord_vr)
    frame_indices = list(range(start_frame_idx, min(start_frame_idx + num_frames * frame_interval, total_frames), frame_interval))

    if len(frame_indices) == 0:
        raise ValueError("No frames selected. Check your start_frame_idx and num_frames.")
    
    if len(frame_indices) < num_frames:
        raise ValueError(f"Requested {num_frames} frames but only {len(frame_indices)} frames are available, please adjust the start_frame_idx and num_frames or decrease the frame_interval.")
        
    video_data = decord_vr.get_batch(frame_indices).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
    return video_data


def get_pixel_values(file_path, num_frames):
    if is_image_file(file_path[0]):
        pixel_values = [open_image(path) for path in file_path]
        pixel_values = [torch.from_numpy(np.array(image)) for image in pixel_values]
        pixel_values = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in pixel_values]
    elif is_video_file(file_path[0]):
        pixel_values = [open_video(video_path, 0, num_frames) for video_path in file_path]
    return pixel_values


class OpenSoraPlanInpaintPipeline(OpenSoraPlanPipeline):

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        super().__init__(vae, text_encoder, tokenizer, scheduler, predict_model, config)
        # If performing continuation or random, the default mask is half of the frame, which can be modified
        self.mask_processor = MaskProcessor(min_clear_ratio=0.5, max_clear_ratio=0.5) 
        self.mask_compressor = MaskCompressor(ae_stride_t=self.vae.vae_scale_factor[0], ae_stride_h=self.vae.vae_scale_factor[1], ae_stride_w=self.vae.vae_scale_factor[2])
        self.noise_adder = None
        if config.add_noise_to_condition:
            self.noise_adder = GaussianNoiseAdder(mean=-4.6, std=0.01, clear_ratio=0)

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    def get_resize_transform(
        self, 
        ori_height,
        ori_width,
        height=None, 
        width=None, 
        crop_for_hw=False, 
        hw_stride=32, 
        max_hxw=236544, # 480 x 480
    ):
        if crop_for_hw:
            assert height is not None and width is not None
            transform = CenterCropResizeVideo((height, width))
        else:
            new_height, new_width = maxhwresize(ori_height, ori_width, max_hxw)
            transform = Compose(
                [
                    CenterCropResizeVideo((new_height, new_width)), # We use CenterCropResizeVideo to share the same height and width, ensuring that the shape of the crop remains consistent when multiple images are captured
                    SpatialStrideCropVideo(stride=hw_stride), 
                ]
            )
        return transform
        
        
    def get_video_transform(self):
        norm_fun = Lambda(lambda x: 2. * x - 1.)
        transform = Compose([
            ToTensorAfterResize(),
            norm_fun
        ])
        return transform

    def get_mask_type_cond_indices(self, mask_type, conditional_pixel_values_path, conditional_pixel_values_indices, num_frames):
        if mask_type is not None and mask_type in STR_TO_TYPE.keys():
            mask_type = STR_TO_TYPE[mask_type]
        if is_image_file(conditional_pixel_values_path[0]):
            if len(conditional_pixel_values_path) == 1:
                mask_type = MaskType.i2v if mask_type is None else mask_type
                if num_frames > 1:
                    conditional_pixel_values_indices = [0] if conditional_pixel_values_indices is None else conditional_pixel_values_indices
                    assert len(conditional_pixel_values_indices) == 1, "conditional_pixel_values_indices should be a list of integers with the same length as conditional_pixel_values_path"
            elif len(conditional_pixel_values_path) == 2:
                mask_type = MaskType.transition if mask_type is None else mask_type
                if num_frames > 1:
                    conditional_pixel_values_indices = [0, -1] if conditional_pixel_values_indices is None else conditional_pixel_values_indices
                    assert len(conditional_pixel_values_indices) == 2, "conditional_pixel_values_indices should be a list of integers with the same length as conditional_pixel_values_path"
            else:
                if num_frames > 1:
                    assert conditional_pixel_values_indices is not None and len(conditional_pixel_values_path) == len(conditional_pixel_values_indices), "conditional_pixel_values_indices should be a list of integers with the same length as conditional_pixel_values_path"
                    mask_type = MaskType.random_temporal if mask_type is None else mask_type
        elif is_video_file(conditional_pixel_values_path[0]):
            # When the input is a video, video continuation is executed by default, with a continuation rate of double
            mask_type = MaskType.continuation if mask_type is None else mask_type
        return mask_type, conditional_pixel_values_indices


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

        if conditional_pixel_values.shape[0] == num_frames:
            inpaint_cond_data = self.mask_processor(conditional_pixel_values, mask_type=mask_type)
            masked_pixel_values, mask = inpaint_cond_data['masked_pixel_values'], inpaint_cond_data['mask']
        else:
            input_pixel_values = torch.zeros([num_frames, 3, height, width], device=device, dtype=weight_dtype)
            input_mask = torch.ones([num_frames, 1, height, width], device=device, dtype=weight_dtype)
            input_pixel_values[conditional_pixel_values_indices] = conditional_pixel_values
            input_mask[conditional_pixel_values_indices] = 0
            masked_pixel_values = input_pixel_values * (input_mask < 0.5)
            mask = input_mask

        print('conditional_pixel_values_indices', conditional_pixel_values_indices)
        print('mask_type', TYPE_TO_STR[mask_type])

        masked_pixel_values = video_transform(masked_pixel_values)

        masked_pixel_values = masked_pixel_values.unsqueeze(0).repeat(batch_size * num_samples_per_prompt, 1, 1, 1, 1).transpose(1, 2).contiguous() # b c t h w
        mask = mask.unsqueeze(0).repeat(batch_size * num_samples_per_prompt, 1, 1, 1, 1).transpose(1, 2).contiguous() # b c t h w

        # add some noise to improve generalization
        if self.noise_adder is not None:
            masked_pixel_values = self.noise_adder(masked_pixel_values, mask)

        masked_pixel_values = masked_pixel_values.to(self.vae.dtype)
        masked_pixel_values = self.vae.encode(masked_pixel_values)

        mask = self.mask_compressor(mask)
    
        masked_pixel_values = torch.cat([masked_pixel_values] * 2) if self.do_classifier_free_guidance else masked_pixel_values
        mask = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask

        masked_pixel_values = masked_pixel_values.to(weight_dtype)
        mask = mask.to(weight_dtype)

        return masked_pixel_values, mask

    @torch.no_grad()
    def __call__(
        self,
        conditional_pixel_values_path: Union[str, List[str]] = None,
        conditional_pixel_values_indices: Union[int, List[int]] = None,
        mask_type: Union[str, MaskType] = None,
        crop_for_hw: bool = False,
        max_hxw: int = 236544,
        prompt: Union[str, List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale: float = 4.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_sequence_length: Optional[int] = 300,
        clean_caption: bool = True,
        enable_temporal_attentions: bool = True,
        added_cond_kwargs: dict = None,
        use_prompt_template: bool = True,
        **kwargs,
    ):

        # 1. Check inputs.
        # text prompt checks
        if use_prompt_template:
            prompt, negative_prompt = self.use_prompt_template(positive_prompt=prompt, negative_prompt=negative_prompt)
        self.text_prompt_checks(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)
        self.generate_params_checks(self.height, self.width)
        self.conditional_pixel_values_checks(conditional_pixel_values_path, conditional_pixel_values_indices, mask_type)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.text_encoder.device or self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0
        self._guidance_scale = guidance_scale

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

        # ==================prepare inpaint data=====================================
        mask_type, conditional_pixel_values_indices = self.get_mask_type_cond_indices(mask_type, conditional_pixel_values_path, conditional_pixel_values_indices, self.num_frames)

        conditional_pixel_values = get_pixel_values(conditional_pixel_values_path, self.num_frames)

        min_height = min([pixels.shape[2] for pixels in conditional_pixel_values])
        min_width = min([pixels.shape[3] for pixels in conditional_pixel_values])

        resize_transform = self.get_resize_transform(
            ori_height=min_height, 
            ori_width=min_width, 
            height=self.height, 
            width=self.width, 
            crop_for_hw=crop_for_hw,
            max_hxw=max_hxw,
        )

        video_transform = self.get_video_transform()
        conditional_pixel_values = torch.cat([resize_transform(pixels) for pixels in conditional_pixel_values])
        real_height, real_width = conditional_pixel_values.shape[-2], conditional_pixel_values.shape[-1]
        # ==================prepare inpaint data=====================================


        # 5. Prepare latents
        latent_channels = self.predict_model.in_channels
        batch_size = batch_size * num_images_per_prompt
        shape = (
            batch_size,
            latent_channels,
            (math.ceil((int(self.num_frames) - 1) / self.vae.vae_scale_factor[0]) + 1) if int(
                self.num_frames) % 2 == 1 else math.ceil(int(self.num_frames) / self.vae.vae_scale_factor[0]),
            math.ceil(int(real_height) / self.vae.vae_scale_factor[1]),
            math.ceil(int(real_width) / self.vae.vae_scale_factor[2]),
        )
        latents = self.prepare_latents(shape, generator=generator, device=device, dtype=prompt_embeds.dtype,
                                       latents=latents)
        # 6 prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ==============================create mask=====================================
        masked_pixel_values, mask = self.get_masked_pixel_values_mask(
            conditional_pixel_values,
            conditional_pixel_values_indices,
            mask_type, 
            batch_size, 
            num_images_per_prompt, 
            self.num_frames, 
            real_height,
            real_width,
            video_transform,
            prompt_embeds.dtype,
            device
        )
        # ==============================create mask=====================================

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
                        "masked_pixel_values": masked_pixel_values,
                        "mask": mask,
                        "return_dict": False}

        latents = self.scheduler.sample(model=self.predict_model, shape=shape, latents=latents, model_kwargs=model_kwargs,
                                        extra_step_kwargs=extra_step_kwargs)
        video = self.decode_latents(latents.to(self.vae.dtype))

        return video


