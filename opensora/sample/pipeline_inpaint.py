
from typing import Callable, List, Optional, Tuple, Union
import math
from matplotlib import widgets
import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

try:
    from opensora.npu_config import npu_config
except:
    npu_config = None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from .pipeline_opensora import EXAMPLE_DOC_STRING, retrieve_timesteps, OpenSoraPipeline

class OpenSoraInpaintPipeline(OpenSoraPipeline):

    def register_image_transforms(self, transforms):
        self.image_transforms = transforms
        print('image transforms register')
    
    def get_masked_video_mask(
        self, 
        conditional_images,
        conditional_images_indices, 
        batch_size, 
        num_images_per_prompt, 
        num_frames, 
        height, 
        width, 
        do_classifier_free_guidance, 
        weight_dtype,
        device
    ):
        # NOTE inpaint
        if isinstance(conditional_images, list) and isinstance(conditional_images[0], torch.Tensor):
            if len(conditional_images[0].shape) == 3:
                conditional_images = [condition_image.unsqueeze(0) for condition_image in conditional_images] # C H W -> 1 C H W
            elif len(conditional_images[0].shape) == 4:
                pass
            conditional_images = torch.cat(conditional_images, dim=0).to(device=device) # F C H W
        elif isinstance(conditional_images, torch.Tensor):
            assert len(conditional_images.shape) == 4, "The shape of conditional_images should be a tensor with 4 dim"
            conditional_images = conditional_images.to(device=device) # F C H W
        else:
            raise NotImplementedError

        input_video = torch.zeros([num_frames, 3, height, width], device=device)
        input_video[conditional_images_indices] = conditional_images.to(input_video.dtype)

        print(conditional_images_indices)

        T, C, H, W = input_video.shape
        mask = torch.ones([T, 1, H, W], device=device)
        mask[conditional_images_indices] = 0
        masked_video = input_video * (mask < 0.5) 

        try:
            masked_video = self.image_transforms(masked_video)
        except:
            raise ValueError("The image_transforms is not defined, please define it first")
        
        masked_video = masked_video.unsqueeze(0).repeat(batch_size * num_images_per_prompt, 1, 1, 1, 1).transpose(1, 2).contiguous() # b c t h w
        mask = mask.unsqueeze(0).repeat(batch_size * num_images_per_prompt, 1, 1, 1, 1).transpose(1, 2).contiguous() # b c t h w
        masked_video = masked_video.to(self.vae.vae.dtype)
        masked_video = self.vae.encode(masked_video)

        # not vae style
        mask = rearrange(mask, 'b c t h w -> (b c t) 1 h w')
        latent_size = (height // self.vae.vae_scale_factor[1], width // self.vae.vae_scale_factor[2])
        if num_frames % 2 == 1:
            latent_size_t = (num_frames - 1) // self.vae.vae_scale_factor[0] + 1
        else:
            latent_size_t = num_frames // self.vae.vae_scale_factor[0]
        mask = F.interpolate(mask, size=latent_size, mode='bilinear')
        mask = rearrange(mask, '(b c t) 1 h w -> b c t h w', t=T, c=1)
        mask_first_frame = mask[:, :, 0:1].repeat(1, 1, self.vae.vae_scale_factor[0], 1, 1)
        mask = torch.cat([mask_first_frame, mask[:, :, 1:]], dim=2).contiguous()
        mask = mask.view(batch_size, latent_size_t, self.vae.vae_scale_factor[0], *latent_size)
        mask = mask.transpose(1, 2).contiguous()
    
        masked_video = torch.cat([masked_video] * 2) if do_classifier_free_guidance else masked_video
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        masked_video = masked_video.to(weight_dtype)
        mask = mask.to(weight_dtype)

        return masked_video, mask

    @torch.no_grad()
    def __call__(
        self,
        # NOTE inpaint
        conditional_images: Optional[List[torch.FloatTensor]] = None,
        conditional_images_indices: Optional[List[int]] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        motion_score: float = None, 
        num_images_per_prompt: Optional[int] = 1,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 300,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
    
        # 1. Check inputs. Raise error if not correct
        num_frames = num_frames or self.transformer.config.sample_size_t * self.vae.vae_scale_factor[0]
        height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]
        self.check_inputs(
            prompt,
            num_frames, 
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # import ipdb;ipdb.set_trace()
        device = getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            num_frames, 
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        masked_video, mask = self.get_masked_video_mask(
            conditional_images, 
            conditional_images_indices, 
            batch_size, 
            num_images_per_prompt, 
            num_frames, 
            height, 
            width,
            do_classifier_free_guidance,
            latents.dtype,
            device
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # inpaint
                latent_model_input = torch.cat([latent_model_input, masked_video, mask], dim=1)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                # import ipdb;ipdb.set_trace()
                if prompt_embeds.ndim == 3:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
                if prompt_attention_mask.ndim == 2:
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
                # prepare attention_mask.
                # b c t h w -> b t h w
                attention_mask = torch.ones_like(latent_model_input)[:, 0]
                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    attention_mask=attention_mask, 
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    motion_score=motion_score, 
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                # print(f'latents_{i}_{t}', torch.max(latents), torch.min(latents), torch.mean(latents), torch.std(latents))
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        # import ipdb;ipdb.set_trace()
        # latents = latents.squeeze(2)
        if not output_type == "latent":
            # b t h w c
            image = self.decode_latents(latents)
            image = image[:, :num_frames, :height, :width]
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

