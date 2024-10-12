import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from megatron.core import InferenceParams
from megatron.core.models.gpt import GPTModel

from .text_encoder.text_encoder import TextEncoder
from .vision.vision_model import VisionModel
from ..data.data_utils.constants import MODEL_CONSTANTS


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

        self.config = config.text_decoder
        self.pre_process = config.pre_process
        self.post_process = config.post_process
        self.add_text_encoder = config.text_encoder is not None
        self.add_image_encoder = config.image_encoder is not None
        self.add_video_encoder = config.video_encoder is not None
        self.add_text_decoder = config.text_decoder is not None
        self.model_constants = MODEL_CONSTANTS.get(config.model_id)
        if self.model_constants:
            self.IGNORE_INDEX = self.model_constants.get("IGNORE_INDEX")
            self.IMAGE_TOKEN_INDEX = self.model_constants.get("IMAGE_TOKEN_INDEX")
        else:
            self.IGNORE_INDEX = None
            self.IMAGE_TOKEN_INDEX = None

        if self.add_text_decoder:
            self.text_decoder = GPTModel(
                config=config.text_decoder,
                transformer_layer_spec=config.text_decoder.language_tansformer_layer_spec,
                vocab_size=config.text_decoder.language_vocab_size,
                max_sequence_length=config.text_decoder.language_max_sequence_length,
                position_embedding_type=config.text_decoder.lm_position_embedding_type,
            )
        if self.add_image_encoder:
            self.image_encoder = VisionModel(
                config.image_encoder,
                config.image_encoder.vision_encoder.vision_transformer_layer_spec,
                config.image_encoder.vision_projector.vision_projection_layer_spec,
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

    def prepare_inputs_labels_for_multimodal(
            self,
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes=None
    ):
        if self.IGNORE_INDEX is None or self.IMAGE_TOKEN_INDEX is None:
            raise AssertionError("IGNORE_INDEX and IMAGE_TOKEN_INDEX shoule be provided for this model.")
        if self.add_image_encoder == False or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.image_encoder(images)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, self.IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.text_decoder.embedding.word_embeddings(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.text_decoder.embedding.word_embeddings(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), self.IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'language_max_sequence_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), self.IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

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

        input_ids, position_ids, attention_mask, past_key_value, combined_embeddings, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, None, labels, images, None
        )

        causal_attention_mask = torch.triu(
            torch.ones(combined_embeddings.shape[0], 1, combined_embeddings.shape[1], combined_embeddings.shape[1],
                       device=combined_embeddings.device),
            diagonal=1
        ).bool()
        attention_mask = ~attention_mask
        expanded_attention_mask = attention_mask[:, None, None, :].expand(
            combined_embeddings.shape[0], 1, combined_embeddings.shape[1], combined_embeddings.shape[1]
        )
        attention_mask = causal_attention_mask.masked_fill(expanded_attention_mask, True)

        outputs = self.text_decoder(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings.transpose(0, 1),
            labels=None,
        )
        logits = outputs.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.text_decoder.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss
