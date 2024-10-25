# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args

from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder

from mindspeed_mm.data.data_utils.mask_utils import MaskCompressor, GaussianNoiseAdder

class SoRAModel(nn.Module):
    """
    Instantiate a video generation model from config.
    SoRAModel is an assembled model, which may include text_encoder, video_encoder, predictor, and diffusion model

    Args:
        config (dict): the general config for Multi-Modal Model
        {
            "ae": {...},
            "text_encoder": {...},
            "predictor": {...},
            "diffusion": {...},
            "load_video_features":False,
            ...
        }
    """

    def __init__(self, config):
        super().__init__()
        self.config = core_transformer_config_from_args(get_args())
        self.load_video_features = config.load_video_features
        self.load_text_features = config.load_text_features
        if not self.load_video_features:
            self.ae = AEModel(config.ae).eval()
            self.ae.requires_grad_(False)
        if not self.load_text_features:
            self.text_encoder = TextEncoder(config.text_encoder).eval()
            self.text_encoder.requires_grad_(False)

        self.predictor = PredictModel(config.predictor).get_model()

        self.is_inpaint_model = True if "Inpaint" in config.predictor.model_id else False
        if self.is_inpaint_model:
            vae_scale_factor = config.ae.vae_scale_factor
            self.mask_compressor = MaskCompressor(ae_stride_t=vae_scale_factor[0], ae_stride_h=vae_scale_factor[1], ae_stride_w=vae_scale_factor[2])
            if config.predictor.add_noise_to_condition:
                self.noise_adder = GaussianNoiseAdder(mean=-3.0, std=0.5, clear_ratio=0.05)
        print(
            f"  Total training parameters = {sum(p.numel() for p in self.predictor.parameters() if p.requires_grad) / 1e9} B")
        self.diffusion = DiffusionModel(config.diffusion).get_model()

    def set_input_tensor(self, input_tensor):
        self.predictor.set_input_tensor(input_tensor)

    def forward(self, video, prompt_ids, video_mask=None, prompt_mask=None, **kwargs):
        """
        video: raw video tensors, or ae encoded latent
        prompt_ids: tokenized input_ids, or encoded hidden states
        video_mask: mask for video/image
        prompt_mask: mask for prompt(text)
        """
        with torch.no_grad():
            # Visual Encode
            if self.load_video_features:
                latents = video
            else:
                if self.is_inpaint_model:
                    video, masked_video, mask = video[:, :3], video[:, 3:6], video[:, 6:7]
                    latents = self.ae.encode(video)
                    if self.noise_adder is not None:
                        masked_latents = self.noise_adder(masked_video, mask)
                    masked_latents = self.ae.encode(masked_video)
                    mask = self.mask_compressor(mask)
                else:
                    latents = self.ae.encode(video)
                
            # Text Encode
            if self.load_text_features:
                prompt = prompt_ids
            else:
                B, N, L = prompt_ids.shape
                prompt_ids = prompt_ids.view(-1, L)
                prompt_mask = prompt_mask.view(-1, L)
                hidden_states = self.text_encoder.encode(prompt_ids, prompt_mask)
                prompt = hidden_states["last_hidden_state"].view(B, N, L, -1)
        
        noised_latents, noise, timesteps = self.diffusion.q_sample(latents, model_kwargs=kwargs, mask=video_mask)

        if self.is_inpaint_model:
            noised_latents = torch.cat([noised_latents, masked_latents, mask], dim=1)

        model_output = self.predictor(
            noised_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt,
            attention_mask=video_mask,
            encoder_attention_mask=prompt_mask,
            **kwargs,
        )
        return model_output, latents, noised_latents, timesteps, noise, video_mask

    def compute_loss(
        self, model_output, latents, noised_latents, timesteps, noise, video_mask
    ):
        """compute diffusion loss"""
        loss_dict = self.diffusion.training_losses(
            model_output=model_output,
            x_start=latents,
            x_t=noised_latents,
            noise=noise,
            t=timesteps,
            mask=video_mask,
        )
        return loss_dict
    
    def train(self, mode=True):
        self.predictor.train()