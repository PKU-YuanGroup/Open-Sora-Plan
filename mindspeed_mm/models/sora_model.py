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

from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.utils.utils import get_device, get_dtype


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
        self.device = get_device(config.device)
        self.dtype = get_dtype(config.dtype)
        self.load_video_features = config.load_video_features
        self.load_text_features = config.load_text_features
        if not self.load_video_features:
            self.ae = AEModel(config.ae).get_model()
        if not self.load_text_features:
            self.text_encoder = TextEncoder(config.text_encoder).get_model()

        self.predictor = PredictModel(config.predictor).get_model()
        self.diffusion = DiffusionModel(config.diffusion).get_model()

    def forward(self, videos, texts, cond_mask, pad_mask, **kwargs):
        """
        videos: raw video tensors, or ae encoded latent
        texts: tokenized input_ids, or encoded hidden states
        cond_mask: mask for texts
        atten_mask: mask for dynamic videos
        """
        with torch.no_grad():
            # Visual Encode
            if self.load_video_features:
                latents = videos.to(self.device, self.dtype)
            else:
                latents = self.ae.encode(videos)
            # Text Encode
            if self.load_text_features:
                cond = texts.to(self.device, self.dtype)
            else:
                cond = self.text_encoder.encode(texts)
        model_args = dict(cond=cond.to(self.device, self.dtype),
                          cond_mask=cond_mask.to(self.device, self.dtype),
                          pad_mask=pad_mask.to(self.device, self.dtype))
        kwargs.update(model_args)

        # compute diffusion loss
        loss_dict = self.diffusion.training_losses(self.predictor, latents, **kwargs)
        return loss_dict
