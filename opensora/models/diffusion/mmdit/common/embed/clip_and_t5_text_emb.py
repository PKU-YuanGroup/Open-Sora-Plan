import numpy
import torch.nn as nn
import transformers
from transformers import T5EncoderModel, T5Tokenizer
from opensora.models.diffusion.mmdit.common.embed.clip_text_emb import AbstractEncoder, FrozenCLIPEmbedder

class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text (from Hugging Face)"""

    def __init__(self, path="google-t5/t5-small", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.transformer = T5EncoderModel.from_pretrained(path)
        self.device = device
        self.max_length = max_length
        self._freeze()

    def _freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        pooled_z = outputs.pooler_output
        return z, pooled_z

    def encode(self, text):
        return self(text)

class TextEmbeddingProjector(nn.Module): # Using instead of Label Embedder
    def __init__(self, clip_in_text_channels, t5_in_text_channels, text_hidden_size, timestep_hidden_size, bias=True):
        self.proj = nn.Linear(clip_in_text_channels + t5_in_text_channels, text_hidden_size, bias=bias)
        self.timestep_proj = nn.Linear(clip_in_text_channels, timestep_hidden_size, bias=bias)

    def forward(self,
                clip_embeds, # may be concat of multiple clip outputs (change the proj too then)
                pooled_clip_embeds, # SDXL takes 0th item as the pooled version
                t5_embeds,
                pooled_t5_embeds): # Byproduct of the encoder above
        # Batch Len Chans
        B, L, C = clip_embeds.shape
        _, _, C_T5 = t5_embeds.shape

        ts_proj = self.timestep_proj(pooled_clip_embeds) # SD3 uses the prompt to influence timestep conditioning

        clip_embeds = nn.functional.pad(clip_embeds, (0, 0, C_T5 - C), mode='constant', value=0)
        embeds = torch.cat([clip_embeds, t5_embeds], dim=1) # concat in L dim
        c = self.proj(embeds)

        return c, ts_proj

class MixedTextEmbedder(nn.Module):
    """
    Embeds text prompt into vector representations. Also handles text dropout for classifier-free guidance.
    """

    def __init__(self, path_clip, path_t5, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.text_encoder_clip = FrozenCLIPEmbedder(path=path_clip)
        self.text_encoder_t5 = FrozenT5Embedder(path=path_t5)
        self.dropout_prob = dropout_prob

        clip_in_text_channels = self.text_encoder_clip.transformer.config.hidden_size # One CLIP model for now
        t5_in_text_channels = self.text_encoder_t5.transformer.config.hidden_size # One CLIP model for now
        self.output_projection = TextEmbeddingProjector(clip_in_text_channels, t5_in_text_channels, hidden_size, hidden_size)

    def token_drop(self, text_prompts, force_drop_ids=None):
        """
        Drops text to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = numpy.random.uniform(0, 1, len(text_prompts)) < self.dropout_prob
        else:
            # TODO
            drop_ids = force_drop_ids == 1
        labels = list(numpy.where(drop_ids, "", text_prompts))
        # print(labels)
        return labels

    def forward(self, text_prompts, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            text_prompts = self.token_drop(text_prompts, force_drop_ids)
        
        embeddings_clip, pooled_embeddings_clip = self.text_encoder_clip(text_prompts)
        embeddings_t5, pooled_embeddings_t5 = self.text_encoder_t5(text_prompts)
        # return embeddings, pooled_embeddings
        text_embeddings, ts_proj = self.output_projection(embeddings_clip, pooled_embeddings_clip, embeddings_t5, pooled_embeddings_t5)
        return text_embeddings, ts_proj
