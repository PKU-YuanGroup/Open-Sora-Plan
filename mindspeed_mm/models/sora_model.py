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
            # TODO: t5固定输入权重情况下如何获取固定输出
            self.text_encoder = TextEncoder(config.text_encoder).eval()
            self.text_encoder.requires_grad_(False)

        self.predictor = PredictModel(config.predictor).get_model()
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

        model_output = self.predictor(
            noised_latents,
            timestep=timesteps,
           # prompt=prompt,
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
