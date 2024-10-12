from typing import List, Optional
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from megatron.core import InferenceParams, mpu
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.text_encoder.text_encoder import TextEncoder
from mindspeed_mm.models.vision.vision_model import VisionModel
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.module_spec.internvl_layer_spec import get_language_layer_spec, get_vit_layer_spec


class InternVLModel(MultiModalModule):
    """
    Vision-Language multi-modal model.
    VLModel is an assembled model, which may include text_encoder, image_encoder, video_encoder, text_decoder model.

    Args:
        config (dict): the general config for VLModel
        {
            "pre_process": (bool),  # Include the embedding leayer in the gpt decoder (used with pipeline parallelism).
            "post_process": (bool),  # Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism).
            "add_text_encoder": (bool),  # Whether to construct the text encoder.
            "add_image_encoder": (bool),  # Whether to construct the image encoder.
            "add_video_encoder": (bool),  # Whether to construct the video encoder.
            "add_text_decoder": (bool),  # Whether to construct the text decoder.
            "img_embedding_idx": (int),  # Index in the language_embeddings tensor where image_embeddings should be inserted.
            "text_encoder": {...},  # Config for the text encoder.
            "image_encoder": {...},  # Config for the image encoder.
            "video_encoder": {...},  # Config for the video encoder.
            "text_decoder": {...},  # Config for the text decoder.
        }
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.config = core_transformer_config_from_args(get_args())
        self.pre_process = config.pre_process
        self.post_process = config.post_process
        self.add_text_encoder = config.text_encoder is not None
        self.add_image_encoder = config.image_encoder is not None
        self.add_video_encoder = config.video_encoder is not None
        self.add_text_decoder = config.text_decoder is not None
        self.img_embedding_idx = config.img_embedding_idx
        self.text_encoder = None
        self.image_encoder = None
        self.video_encoder = None
        self.text_decoder = None

        self.vocab_size = config.text_decoder.vocab_size
        self.img_context_token_id = config.img_context_token_id
        
        # initialize pipeline prarallel configs
        self.pp_size = mpu.get_pipeline_model_parallel_world_size()
        if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
            raise NotImplementedError("Not support virtual_pipeline_model_parallel now")
        else:
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        
        #  This attribute is needed to check if an all-reduce is required
        #  on the word embeddings inside 'finalize_model_grads._allreduce_word_embedding_grads'.
        self.share_embeddings_and_output_weights = False
        if self.add_text_decoder:
            self.text_decoder = self._build_text_decoder_model(config.text_decoder)
            if self.text_decoder is not None:
                self.share_embeddings_and_output_weights = self.text_decoder.share_embeddings_and_output_weights
            else:
                self.add_text_decoder = False
        if self.add_text_encoder:
            self.text_encoder = TextEncoder(config.text_encoder).get_model()
        if self.add_image_encoder:
            vit_hidden_size = config.image_encoder.vision_encoder.hidden_size
            llm_hidden_size = config.text_decoder.hidden_size
            self.downsample_ratio = config.downsample_ratio

            self.vit_proj = nn.Sequential(
                nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
                nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )            
            self.image_encoder = VisionModel(
                config.image_encoder,
                encoder_transformer_layer_spec=get_vit_layer_spec(config.image_encoder.vision_encoder))
        if self.add_video_encoder:
            raise NotImplementedError("Not support video_encoder now")

    def _build_text_decoder_model(self, config):
        language_transformer_layer_spec = get_language_layer_spec()
        if self.pp_size <= 1:
            return GPTModel(
                config=config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                rotary_percent=config.rotary_percent,
                rotary_base=config.rotary_base,
                pre_process=self.pre_process,
                post_process=self.post_process,
                fp16_lm_cross_entropy=config.fp16_lm_cross_entropy
            )
        if self.pp_size != len(config.pipeline_layer_index):
            raise ValueError(f"length of pipeline_layer_index must equal to pipeline-model-parallel-size, "
                             f"but got pipeline_layer_index length:{len(config.pipeline_layer_index)} "
                             f"and pipeline-model-parallel-size:{self.pp_size}.")
        pipeline_start_index = config.pipeline_layer_index[self.pp_rank]
        if mpu.is_pipeline_last_stage():
            pipeline_end_index = config.num_layers
        else:
            pipeline_end_index = config.pipeline_layer_index[self.pp_rank + 1]
        
        if pipeline_end_index < pipeline_start_index:
            raise ValueError(f"each index in pipeline_layer_index must equal or large than last, "
                             f"but got {pipeline_end_index} and {pipeline_start_index}.")
        if pipeline_end_index - pipeline_start_index > 0:
            config.num_layers = pipeline_end_index - pipeline_start_index
        else:
            return None

        if not mpu.is_pipeline_first_stage():
            self.pre_process = False
            self.add_text_encoder = False
            self.add_image_encoder = False
            self.add_video_encoder = False
        
        if not mpu.is_pipeline_last_stage():
            self.post_process = False
        
        print(f'text decoder pipeline config:\
              pp_rank:{self.pp_rank},\
              pre_process:{self.pre_process},\
              post_process:{self.post_process},\
              num_layers:{config.num_layers}')
        # GPTModel will divide num_layers by pp_size
        config.num_layers *= self.pp_size
        model = GPTModel(
                config=config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                rotary_percent=config.rotary_percent,
                rotary_base=config.rotary_base,
                pre_process=self.pre_process,
                post_process=self.post_process,
                fp16_lm_cross_entropy=config.fp16_lm_cross_entropy
            )
        config.num_layers //= self.pp_size
        return model

    def shared_embedding_or_output_weight(self):
        """
        This is a convenience method to surface the language model's word embeddings, which is 
        necessary for 'finalize_model_grads._allreduce_word_embedding_grads'.
        """
        if self.add_text_decoder:
            return self.text_decoder.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if not len(input_tensor) == 1:
            raise AssertionError("input_tensor should only be length 1 for vlmodel")
        if self.add_image_encoder:
            self.image_encoder.set_input_tensor(input_tensor[0])
        elif self.add_text_decoder:
            self.text_decoder.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_text_decoder: bool = False,
        freeze_image_encoder: bool = False,
        freeze_image_projection: bool = False,
        freeze_video_encoder: bool = False
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_text_decoder (bool): Freeze the text decoder module.
            freeze_image_encoder (bool): Freeze the image encoder module.
            freeze_image_projection (bool): Freeze the image projector module.
            freeze_video_encoder (bool): Freeze the video encoder module.
        """
        if freeze_text_decoder and self.text_decoder is not None:
            for param in self.text_decoder.parameters():
                param.requires_grad = False
        self.image_encoder.freeze(freeze_image_encoder, freeze_image_projection)

    def compute_loss(self, logits, labels, ignore_flag=False):
        # 偏移tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if ignore_flag:
            loss = loss * 0.0

        return loss
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        if self.add_image_encoder:
            image_flags = image_flags.squeeze(-1)
            vit_embeds = self.image_encoder(image)
            vit_embeds = self.vit_proj(vit_embeds)
            vit_embeds = vit_embeds[image_flags == 1]
            vit_batch_size = image.shape[0]
            B = input_ids.shape[0]
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}')
            if not self.add_text_decoder:
                return vit_embeds

        if self.add_text_decoder:
            input_embeds = None
            if self.pre_process:
                input_embeds = self.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()
                input_embeds = input_embeds.transpose(0, 1)
                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)
                print(f'dynamic token length: {N}')
                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == self.img_context_token_id)
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                input_embeds = input_embeds.reshape(B, N, C).transpose(0, 1)

            outputs = self.text_decoder(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=input_embeds,
                labels=None
            )
            
            if self.post_process:
                logits = outputs
                logits = logits.float()

                loss = None
                if labels is not None:
                    loss = self.compute_loss(logits, labels)
                    
                return {
                    "loss": loss,
                    "logits": logits
                }
            else:
                return outputs
