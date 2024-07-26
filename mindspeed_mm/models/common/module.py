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


from megatron.core import mpu
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class MultiModalModule(MegatronModule):

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

    def set_input_tensor(self, input_tensor):
        """
        Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func
        """
        self.input_tensor = input_tensor

    def build_layer(self, args):
        """
        Build model layers for each pipeline groups.
        """
        raise NotImplementedError("build_layer function must be implemented")

    def _get_num_layers(self, layer_number):
        """
        Get model layers number for each pipeline groups.
        """
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size > 1:
            if layer_number % pp_size != 0:
                raise AssertionError(
                    "num_layers (%d) must be divisible by number of "
                    "pipeline_model_parallel_world_size (%d)"
                    % (layer_number, pp_size)
                )
            return layer_number // pp_size
        else:
            return layer_number

    def _get_layer(self, layer_number):
        """
        Get model layers.
        """
        return self.layers[layer_number]
