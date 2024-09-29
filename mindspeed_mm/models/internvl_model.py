from typing import List, Optional
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from megatron.core import InferenceParams
from megatron.core.models.gpt import GPTModel

from mindspeed_mm.models.text_encoder.text_encoder import TextEncoder
from mindspeed_mm.models.vision.vision_model import VisionModel


class InternVLModel(nn.Module):
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
        super().__init__()

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
        self.downsample_ratio = config.downsample_ratio
        self.img_context_token_id = config.img_context_token_id
        
        #  This attribute is needed to check if an all-reduce is required
        #  on the word embeddings inside 'finalize_model_grads._allreduce_word_embedding_grads'.
        self.share_embeddings_and_output_weights = False
        if self.add_image_encoder:
            self.image_encoder = VisionModel(config.image_encoder)
        if self.add_text_decoder:
            self.text_decoder = GPTModel(
                config=config.text_decoder,
                transformer_layer_spec=config.text_decoder.language_tansformer_layer_spec,
                vocab_size=config.text_decoder.vocab_size,
                max_sequence_length=config.text_decoder.max_position_embeddings,
                parallel_output=config.text_decoder.parallel_output,
                position_embedding_type=config.text_decoder.position_embedding_type,
                rotary_percent=config.text_decoder.rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=config.text_decoder.rotary_base,
                fp16_lm_cross_entropy=config.text_decoder.fp16_lm_cross_entropy
            )
            self.share_embeddings_and_output_weights = self.text_decoder.share_embeddings_and_output_weights
        if self.add_text_encoder:
            self.text_encoder = TextEncoder(config.text_encoder).get_model()
        if self.add_video_encoder:
            raise NotImplementedError("video_encoder module has not been implemented")
        
        vit_hidden_size = config.image_encoder.vision_encoder.hidden_size
        llm_hidden_size = config.text_decoder.hidden_size

        self.vit_proj = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

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
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
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

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()
        input_embeds = input_embeds.transpose(0, 1)

        vit_embeds = self.image_encoder(image)
        vit_embeds = self.vit_proj(vit_embeds)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = image.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C).transpose(0, 1)

        outputs = self.text_decoder(
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            decoder_input=input_embeds,
            labels=None
        )
        logits = outputs
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, ignore_flag)
            
        return {
            "loss": loss,
            "logits": logits
        }
    