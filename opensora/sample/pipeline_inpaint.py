
import inspect
import os
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from altair import condition
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import decord

from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda, Resize

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, HunyuanDiT2DModel
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import logging, BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from opensora.models.diffusion.opensora_v1_3.modeling_inpaint import OpenSoraInpaint_v1_3
from opensora.sample.pipeline_opensora import OpenSoraPipeline, OpenSoraPipelineOutput, rescale_noise_cfg
from opensora.dataset.transform import CenterCropResizeVideo, SpatialStrideCropVideo,ToTensorAfterResize, maxhwresize
from opensora.utils.mask_utils import MaskProcessor, MaskCompressor, GaussianNoiseAdder, MaskType, STR_TO_TYPE, TYPE_TO_STR

try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

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


class OpenSoraInpaintPipeline(OpenSoraPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: MT5Tokenizer,
        transformer: OpenSoraInpaint_v1_3,
        scheduler: DDPMScheduler,
        text_encoder_2: CLIPTextModelWithProjection = None,
        tokenizer_2: CLIPTokenizer = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
        )
        
        # If performing continuation or random, the default mask is half of the frame, which can be modified
        self.mask_processor = MaskProcessor(min_clear_ratio=0.5, max_clear_ratio=0.5) 

        self.mask_compressor = MaskCompressor(ae_stride_t=self.vae.vae_scale_factor[0], ae_stride_h=self.vae.vae_scale_factor[1], ae_stride_w=self.vae.vae_scale_factor[2])
        
        self.noise_adder = None

    def check_inputs(
        self,
        conditional_pixel_values_path,
        conditional_pixel_values_indices,
        mask_type,
        max_hxw,
        noise_strength,
        prompt,
        num_frames,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        prompt_embeds_2=None,
        negative_prompt_embeds_2=None,
        prompt_attention_mask_2=None,
        negative_prompt_attention_mask_2=None,
        callback_on_step_end_tensor_inputs=None,
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
        
        if not isinstance(max_hxw, int) or not (max_hxw >= 102400 and max_hxw <= 236544):
            raise  ValueError("max_hxw should be an integer between 102400 and 236544")
        
        if not isinstance(noise_strength, float) or not (noise_strength >= 0 and noise_strength <= 1):
            raise ValueError("noise_strength should be a non-negative float")
        
        super().check_inputs(
            prompt,
            num_frames,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )

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
        
        if self.noise_adder is not None:
            # add some noise to improve motion strength
            masked_pixel_values = self.noise_adder(masked_pixel_values, mask)
        
        masked_pixel_values = masked_pixel_values.to(self.vae.vae.dtype)
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
        noise_strength: Optional[float] = 0.0,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_samples_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attention_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        max_sequence_length: int = 512,
        device = None, 
    ):
        
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        num_frames = num_frames or (self.transformer.config.sample_size_t - 1) * self.vae.vae_scale_factor[0] + 1
        height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            conditional_pixel_values_path,
            conditional_pixel_values_indices,
            mask_type,
            max_hxw,
            noise_strength,
            prompt,
            num_frames, 
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = device or getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')


        # 3. Encode input prompt

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=self.transformer.dtype,
            num_samples_per_prompt=num_samples_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            text_encoder_index=0,
        )
        if self.tokenizer_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_attention_mask_2,
                negative_prompt_attention_mask_2,
            ) = self.encode_prompt(
                prompt=prompt,
                device=device,
                dtype=self.transformer.dtype,
                num_samples_per_prompt=num_samples_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds_2,
                negative_prompt_embeds=negative_prompt_embeds_2,
                prompt_attention_mask=prompt_attention_mask_2,
                negative_prompt_attention_mask=negative_prompt_attention_mask_2,
                max_sequence_length=77,
                text_encoder_index=1,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_attention_mask_2 = None
            negative_prompt_attention_mask_2 = None

        # ==================prepare inpaint data=====================================
        if noise_strength != 0:
            self.noise_adder = GaussianNoiseAdder(mean=np.log(noise_strength), std=0.01, clear_ratio=0)

        mask_type, conditional_pixel_values_indices = self.get_mask_type_cond_indices(mask_type, conditional_pixel_values_path, conditional_pixel_values_indices, num_frames)

        conditional_pixel_values = get_pixel_values(conditional_pixel_values_path, num_frames)

        min_height = min([pixels.shape[2] for pixels in conditional_pixel_values])
        min_width = min([pixels.shape[3] for pixels in conditional_pixel_values])

        resize_transform = self.get_resize_transform(
            ori_height=min_height, 
            ori_width=min_width, 
            height=height, 
            width=width, 
            crop_for_hw=crop_for_hw,
            max_hxw=max_hxw,
        )

        video_transform = self.get_video_transform()
        conditional_pixel_values = torch.cat([resize_transform(pixels) for pixels in conditional_pixel_values])
        real_height, real_width = conditional_pixel_values.shape[-2], conditional_pixel_values.shape[-1]
        # ==================prepare inpaint data=====================================
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        if get_sequence_parallel_state():
            world_size = hccl_info.world_size if torch_npu is not None else nccl_info.world_size
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_samples_per_prompt,
            num_channels_latents,
            (num_frames + world_size - 1) // world_size if get_sequence_parallel_state() else num_frames, 
            real_height,
            real_width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ==============================create mask=====================================
        masked_pixel_values, mask = self.get_masked_pixel_values_mask(
            conditional_pixel_values,
            conditional_pixel_values_indices,
            mask_type, 
            batch_size, 
            num_samples_per_prompt, 
            num_frames, 
            real_height,
            real_width,
            video_transform,
            prompt_embeds.dtype,
            device
        )
        # ==============================create mask=====================================

        # 7 create image_rotary_emb, style embedding & time ids
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
            if self.tokenizer_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
                prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        if self.tokenizer_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(device=device)
            prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)

        # ==================make sp=====================================
        if get_sequence_parallel_state():
            prompt_embeds = rearrange(
                prompt_embeds, 
                'b (n x) h -> b n x h', 
                n=world_size,
                x=prompt_embeds.shape[1] // world_size
                ).contiguous()
            rank = hccl_info.rank if torch_npu is not None else nccl_info.rank
            prompt_embeds = prompt_embeds[:, rank, :, :]

            latents_num_frames = latents.shape[2]
            masked_pixel_values = masked_pixel_values[:, :, latents_num_frames * rank: latents_num_frames * (rank + 1)]
            mask = mask[:, :, latents_num_frames * rank: latents_num_frames * (rank + 1)]
        # ==================make sp=====================================

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # inpaint
                latent_model_input = torch.cat([latent_model_input, masked_pixel_values, mask], dim=1)

                # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                t_expand = torch.tensor([t] * latent_model_input.shape[0], device=device).to(
                    dtype=latent_model_input.dtype
                )

                # ==================prepare my shape=====================================
                # predict the noise residual
                if prompt_embeds.ndim == 3:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
                if prompt_attention_mask.ndim == 2:
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
                if prompt_embeds_2 is not None and prompt_embeds_2.ndim == 2:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d
                
                attention_mask = torch.ones_like(latent_model_input)[:, 0].to(device=device)
                # ==================prepare my shape=====================================

                # ==================make sp=====================================
                if get_sequence_parallel_state():
                    attention_mask = attention_mask.repeat(1, world_size, 1, 1)
                # ==================make sp=====================================

                noise_pred = self.transformer(
                    latent_model_input,
                    attention_mask=attention_mask, 
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=t_expand,
                    pooled_projections=prompt_embeds_2,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    prompt_embeds_2 = callback_outputs.pop("prompt_embeds_2", prompt_embeds_2)
                    negative_prompt_embeds_2 = callback_outputs.pop(
                        "negative_prompt_embeds_2", negative_prompt_embeds_2
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # ==================make sp=====================================
        if get_sequence_parallel_state():
            latents_shape = list(latents.shape)  # b c t//sp h w
            full_shape = [latents_shape[0] * world_size] + latents_shape[1:]  # # b*sp c t//sp h w
            all_latents = torch.zeros(full_shape, dtype=latents.dtype, device=latents.device)
            torch.distributed.all_gather_into_tensor(all_latents, latents)
            latents_list = list(all_latents.chunk(world_size, dim=0))
            latents = torch.cat(latents_list, dim=2)
        # ==================make sp=====================================

        if not output_type == "latent":
            videos = self.decode_latents(latents)
            videos = videos[:, :num_frames, :real_height, :real_width]
        else:
            videos = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (videos, )

        return OpenSoraPipelineOutput(videos=videos)
