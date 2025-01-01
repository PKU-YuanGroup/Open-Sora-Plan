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

import os
import glob
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Union

from .vae import VideoAutoencoderKL, VideoAutoencoder3D
from .casualvae import CausalVAE
from .wfvae import WFVAE

from mindspeed_mm.utils.utils import get_dtype

AE_MODEL_MAPPINGS = {"vae": VideoAutoencoderKL,
                     "vae3D": VideoAutoencoder3D,
                     "casualvae": CausalVAE,
                     "wfvae": WFVAE}


class AEModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = get_dtype(config.dtype)
        self.model = AE_MODEL_MAPPINGS[config.model_id](**config.to_dict()).to(self.dtype)
        self.register_buffer('shift', torch.tensor(config.shift, dtype=self.dtype)[None, :, None, None, None])
        self.register_buffer('scale', torch.tensor(config.scale, dtype=self.dtype)[None, :, None, None, None])

    def get_model(self):
        return self.model

    def encode(self, x):
        x = (self.model.encode(x).sample() - self.shift.to(x.device, dtype=x.dtype)) * self.scale.to(x.device, dtype=x.dtype)
        return x

    def decode(self, x):
        x = x / self.scale.to(x.device, dtype=x.dtype) + self.shift.to(x.device, dtype=x.dtype)
        x = self.model.decode(x)
        return x

    def forward(self, x):
        raise NotImplementedError("forward function is not implemented")