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
import torch.nn as nn
from mindspeed_mm.model.ae import AEModel
from mindspeed_mm.model.diffusion import DiffusionModel
from mindspeed_mm.model.predictor import PredictModel
from mindspeed_mm.model.text_encoder import TextEncoder
from mindspeed_mm.model.utils.utils import get_device


class SDModel(nn.Module):
    """
    SD Model is design for Text to Image Generation Task
    And it follows diffusers and dit as example

    Args:
        config (dict): THe general config for Text to Image Generation Model such as SDXL/SD3 and Dit
        {
            "text_encoder" : {...} # Config for textencoder
            "text_encoder_1" : {...} # Config for textencoder1
            "text_encoder_2" : {...} # Config for textencoder2
            "predictor" : {...} # Config for predictor
            "autoencoder" : {...} # Config for autoencoder
            "diffusion" : {...} # Config for

        }

    """

    def __init__(
        self,
        config,
    ):
        super.__init__()
        self.config = config

        self.predictor = PredictModel(self.config.predictor)
        self.diffusion = DiffusionModel(self.config.predictor)

        if not self.config.load_image_feature:
            self.ae = AEModel(self.config.ae)
        self.freezen_model = []

        if "train_autoencoder" not in self.config.keys():
            self.freezen_model.append(self.ae)

        if not self.config.load_text_feature:
            self.text_encoders = []
            if "text_encoder_list" in self.config.keys():
                for text_encoder in self.config["text_encoder_list"]:
                    self.text_encoders.append(
                        TextEncoder(self.config["text_encoder_list"][text_encoder])
                    )
            else:
                self.text_encoders.append(TextEncoder(self.config["text_encoder"]))

        for module in self.freezen_model:
            module.requires_grad = False

        self.device = get_device(self.config.device)

    def forward(
        self,
        image,
        text,
        **kwargs,
    ):
        x = image.to(self.device)
        text = text.to(self.device)

        # handle the image
        if not self.config.load_image_feature:
            x = self.ae.encode(x).latent_dist.sample().mul_(self.config["ae"].ae_factor)

        # handle the text
        if ("text_encoder" in self.config.keys()) and (
            not self.config.load_text_feature
        ):
            # uning SDXL/SD3 as example
            prompts_embeds = []
            for text_encoder in self.text_encoders:
                prompts_embed = text_encoder(
                    text, output_hidden_states=True, retrun_dict=False
                )
                pooled_embed = prompts_embed[0]
                prompts_embed = prompts_embed[-1][-2]
                bs_text, seq_len, _ = prompts_embed.shape
                prompts_embed = prompts_embed.view(bs_text, seq_len, -1)
                prompts_embeds.append(prompts_embed)

            prompts_embed = torch.concat(
                prompts_embeds, dim=-1
            )  # 768 1280 2048 -> 4096
            pooled_embed = pooled_embed.view(bs_text, -1)
            model_kwargs = {
                "prompts_embeds": prompts_embed,
                "pooled_embeds": pooled_embed,
            }
        else:
            # dit using caption label
            model_kwargs = dict(text=text)

        # get timestep
        t = torch.randint(
            0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device
        )

        # compute loss
        loss_dict = self.diffusion.training_losses(self.predictor, x, t, model_kwargs)
        return loss_dict
