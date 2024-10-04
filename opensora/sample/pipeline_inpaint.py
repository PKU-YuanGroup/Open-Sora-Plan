
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

from opensora.models.diffusion.opensora_v1_2.modeling_inpaint import OpenSoraInpaint_v1_2
from opensora.sample.pipeline_opensora import OpenSoraPipeline, OpenSoraPipelineOutput, rescale_noise_cfg
from opensora.dataset.transform import CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, ToTensorAfterResize
from opensora.utils.mask_utils import MaskProcessor, MaskCompressor, MaskType, STR_TO_TYPE

try:
    import torch_npu
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
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
    try:
        image = Image.open(file_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Failed to open image: {e}")
        return None

def open_video(file_path, start_frame_idx, num_frames, frame_interval=):
    # 尝试使用 decord 打开视频
    try:
        decord_vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=1)


        
        video_data = decord_vr.get_batch(frame_indices).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data
    except Exception as e:
        print(f"Failed to open video: {e}")
        return None

class OpenSoraInpaintPipeline(OpenSoraPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: MT5Tokenizer,
        transformer: OpenSoraInpaint_v1_2,
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
        
        self.mask_processor = MaskProcessor()

        self.mask_compressor = MaskCompressor(ae_stride_t=self.vae.vae_scale_factor[0], ae_stride_h=self.vae.vae_scale_factor[1], ae_stride_w=self.vae.vae_scale_factor[2])
        
    def check_inputs(
        self,
        conditional_pixel_values_path,
        mask_type,
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
        if not isinstance(conditional_pixel_values_path, str) and not isinstance(conditional_pixel_values_path, list):
            raise ValueError("conditional_pixel_values_path should be a string or a list of strings")
            
        
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
        ori_height=None, 
        ori_width=None, 
        height=None, 
        width=None, 
        crop_for_hw=False, 
        hw_stride=32, 
        max_hw_square=1024 * 1024,
    ):
        if crop_for_hw:
            assert height is not None and width is not None
            transform = CenterCropResizeVideo((height, width))
        else:
            if ori_height * ori_width > max_hw_square:
                scale_factor = np.sqrt(max_hw_square / (ori_height * ori_width))
                new_height = int(ori_height * scale_factor)
                new_width = int(ori_width * scale_factor)
            else:
                new_height = ori_height
                new_width = ori_width
            transform = Compose(
                [
                    CenterCropResizeVideo((new_height, new_width)),
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

    
    def get_masked_video_mask(
        self, 
        conditional_images,
        conditional_images_indices, 
        batch_size, 
        num_samples_per_prompt, 
        num_frames, 
        height, 
        width, 
        video_transform,
        weight_dtype,
        device
    ):
        
        input_video = torch.zeros([num_frames, 3, height, width], device=device)
        input_video[conditional_images_indices] = conditional_images.to(device=device, dtype=input_video.dtype)

        print(conditional_images_indices)

        T, C, H, W = input_video.shape
        mask = torch.ones([T, 1, H, W], device=device)
        mask[conditional_images_indices] = 0
        masked_video = input_video * (mask < 0.5) 

        masked_video = video_transform(masked_video)

        masked_video = masked_video.unsqueeze(0).repeat(batch_size * num_samples_per_prompt, 1, 1, 1, 1).transpose(1, 2).contiguous() # b c t h w
        mask = mask.unsqueeze(0).repeat(batch_size * num_samples_per_prompt, 1, 1, 1, 1).transpose(1, 2).contiguous() # b c t h w
        masked_video = masked_video.to(self.vae.vae.dtype)
        masked_video = self.vae.encode(masked_video)

        mask = self.mask_compressor(mask)
    
        masked_video = torch.cat([masked_video] * 2) if self.do_classifier_free_guidance else masked_video
        mask = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask

        masked_video = masked_video.to(weight_dtype)
        mask = mask.to(weight_dtype)

        return masked_video, mask
    
    @torch.no_grad()
    def __call__(
        self,
        conditional_pixel_values_path: Union[str, List[str]] = None,
        mask_type: Union[str, MaskType] = MaskType.t2iv,
        crop_for_hw: bool = False,
        max_hw_square: int = 1024 * 1024,
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
        motion_score: float = None, 
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
            mask_type,
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
        assert conditional_images is not None, "conditional_images should not be None"
        assert conditional_images_indices is None \
            or (isinstance(conditional_images_indices, list) and isinstance(conditional_images_indices[0], int)) 
            
        if isinstance(conditional_images, str):
            conditional_images = [conditional_images]
            conditional_images_indices = [0] if conditional_images_indices is None or len(conditional_images_indices) != 1 else conditional_images_indices
        elif isinstance(conditional_images, list) and isinstance(conditional_images[0], str):
            if len(conditional_images) == 1:
                conditional_images_indices = [0] if conditional_images_indices is None or len(conditional_images_indices) != 1 else conditional_images_indices
            elif len(conditional_images) == 2:
                conditional_images_indices = [0, -1] if conditional_images_indices is None or len(conditional_images_indices) != 2 else conditional_images_indices
            else:
                assert conditional_images_indices is not None and len(conditional_images) == len(conditional_images_indices)
        else:
            raise ValueError("conditional_images should be a str or a list of str")
        

        conditional_images = [Image.open(image).convert("RGB") for image in conditional_images]
        conditional_images = [torch.from_numpy(np.copy(np.array(image))) for image in conditional_images]
        conditional_images = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in conditional_images]

        min_height = min([image.shape[2] for image in conditional_images])
        min_width = min([image.shape[3] for image in conditional_images])

        resize_transform = self.get_resize_transform(
            ori_height=min_height, 
            ori_width=min_width, 
            height=height, 
            width=width, 
            crop_for_hw=crop_for_hw,
            max_hw_square=max_hw_square,
        )

        video_transform = self.get_video_transform()
        conditional_images = torch.cat([resize_transform(image) for image in conditional_images])
        real_height, real_width = conditional_images.shape[-2], conditional_images.shape[-1]
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

        print('latents', latents.shape)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        masked_video, mask = self.get_masked_video_mask(
            conditional_images,
            conditional_images_indices, 
            batch_size, 
            num_samples_per_prompt, 
            num_frames, 
            real_height,
            real_width,
            video_transform,
            prompt_embeds.dtype,
            device
        )

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
            masked_video = masked_video[:, :, latents_num_frames * rank: latents_num_frames * (rank + 1)]
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
                latent_model_input = torch.cat([latent_model_input, masked_video, mask], dim=1)

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
                motion_score_tensor = None
                if motion_score is not None:
                    motion_score_tensor = torch.tensor([motion_score] * latent_model_input.shape[0]).to(device=device)
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
                    motion_score=motion_score_tensor, 
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