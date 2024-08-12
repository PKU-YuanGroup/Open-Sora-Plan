# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import nn

from megatron.core import InferenceParams
from megatron.core.models.gpt import GPTModel

from .text_encoder.text_encoder import TextEncoder
from .vision.vision_model import VisionModel


class VLModel(nn.Module):
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

        #  This attribute is needed to check if an all-reduce is required
        #  on the word embeddings inside 'finalize_model_grads._allreduce_word_embedding_grads'.
        self.share_embeddings_and_output_weights = False
        if self.add_text_decoder:
            self.text_decoder = GPTModel(
                config=config.text_decoder,
                transformer_layer_spec=config.language_tansformer_layer_spec,
                vocab_size=config.language_vocab_size,
                max_sequence_length=config.language_max_sequence_length,
                parallel_output=config.parallel_output,
                position_embedding_type=config.language_position_embedding_type,
                rotary_percent=config.language_rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=config.language_rotary_base
            )
            self.share_embeddings_and_output_weights = self.text_decoder.share_embeddings_and_output_weights
        if self.add_image_encoder:
            self.image_encoder = VisionModel(config.image_encoder)
        if self.add_text_encoder:
            self.text_encoder = TextEncoder(config.text_encoder).get_model()
        if self.add_video_encoder:
            # TODO: video_encoder needs to be implemented
            raise NotImplementedError("video_encoder module has not been implemented")

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
        # TODO: freeze function of VideoEncoder needs to be implemented

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None
    ) -> torch.Tensor:
        """
        Forward function of the VLModel.

        Args:
            images (torch.Tensor): Input image of shape [batch, img_h, img_w].
            input_ids (torch.Tensor): Input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): Input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Attention mask for the text decoder model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference parameter for the forward method of GPTModel.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        """

        if self.add_image_encoder:
            image_embeddings = self.image_encoder(images)
        else:
            image_embeddings = None

        if not self.add_text_decoder:
            return image_embeddings

        if self.pre_process:
            language_embeddings = self.text_decoder.embedding(
                input_ids=input_ids, position_ids=position_ids
            )  # [text_seq_len, b, h_language]

            # If running inference, we can skip image token computation if they were computed already earlier for this sample.

            combined_embeddings = torch.cat(
                [
                    language_embeddings[: self.img_embedding_idx],
                    image_embeddings,
                    language_embeddings[self.img_embedding_idx :],
                ],
                dim=0,
            )  # [combined_seq_len, b, h_language]
        else:
            combined_embeddings = None

        output = self.text_decoder(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params,
        )

        return output
    