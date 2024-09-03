# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


from megatron.core import mpu
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class MultiModalModule(MegatronModule):

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.input_tensor = None

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

    def build_layer(self, *args, **kwargs):
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
                    "pipeline_model_parallel_world_size (%d)" % (layer_number, pp_size)
                )
            return layer_number // pp_size
        else:
            return layer_number

    def _get_layer(self, layer_number):
        """
        Get model layers.
        """
        return self.layers[layer_number]
