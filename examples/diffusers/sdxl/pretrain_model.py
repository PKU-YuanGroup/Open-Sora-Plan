import math
import os
from typing import Optional

import torch
import torch_npu
from torch import nn
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    if len(x.shape) != 2:
        raise ValueError("timestep embedding has to be 2 dimensions")
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection,
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int,
):
    r"""
    workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
    instead of the hidden states for the EOS token
    If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

    Original code from CLIP's pooling function:

    \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
    \# take features from the eot embedding (eot_token is the highest number in each sequence)
    \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]
    """

    # input_ids: b*n,77
    # find index for EOS token

    # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
    # eos_token_index = torch.where(input_ids == eos_token_id)[1]
    # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # Use argmax to find the last index of the EOS token for each element in the batch
    eos_token_index = torch.argmax(eos_token_mask, dim=1).to(
        device=last_hidden_state.device
    )  # this will be 0 if there is no EOS token, it's fine

    # get hidden states for EOS token
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        eos_token_index,
    ]

    # apply projection: projection may be of different dtype than last_hidden_state
    pooled_output = text_encoder.text_projection(
        pooled_output.to(text_encoder.text_projection.weight.dtype)
    ).to(dtype=last_hidden_state.dtype, device=last_hidden_state.device)

    return pooled_output


def get_hidden_states_sdxl(
    max_token_length: int,
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
    weight_dtype: Optional[str] = None,
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape(
        (-1, tokenizer1.model_max_length)
    )  # batch_size*n, 77
    input_ids2 = input_ids2.reshape(
        (-1, tokenizer2.model_max_length)
    )  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    pool2 = pool_workaround(
        text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id
    )

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = (
        1
        if max_token_length is None
        else max_token_length // (tokenizer1.model_max_length - 2)
    )
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(
                hidden_states1[:, i : i + tokenizer1.model_max_length - 2]
            )  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[
                :, i : i + tokenizer2.model_max_length - 2
            ]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(
            hidden_states2[:, -1].unsqueeze(1)
        )  # <EOS> か <PAD> のどちらか
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents, epoch, step, weight_dtype
):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0
    max_timestep = noise_scheduler.config.num_train_timesteps

    timesteps = torch.randint(
        min_timestep, max_timestep, (b_size,), device=latents.device
    ).long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(
        weight_dtype
    )

    return noise, noisy_latents, timesteps


class SdxlPretrainModels(nn.Module):
    def __init__(
        self,
        args,
        unet: nn.Module,
        text_encoder1: nn.Module,
        text_encoder2: nn.Module,
        weight_dtype,
    ):
        super().__init__()
        self.args = args
        self.unet = unet
        self.text_encoder1 = text_encoder1
        self.text_encoder2 = text_encoder2
        self.weight_dtype = weight_dtype

    def forward(
        self,
        batch,
        accelerator,
        noise_scheduler,
        latent,
        epoch,
        step,
        encoder_hidden_state1,
        encoder_hidden_state2,
        pool2,
    ):
        with torch.set_grad_enabled(True):

            def compute_time_ids(original_size, crops_coords_top_left):
                target_size = (self.args.resolution, self.args.resolution)
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids])
                add_time_ids = add_time_ids.to(
                    accelerator.device, dtype=self.weight_dtype
                )
                return add_time_ids

            # get size embeddings
            orig_size = batch["original_sizes_hw"]
            crop_size = batch["crop_top_lefts"]
            target_size = batch["target_sizes_hw"]

            embs = get_size_embeddings(
                orig_size, crop_size, target_size, accelerator.device
            ).to(device=accelerator.device, dtype=self.weight_dtype)
            time_ids = []
            for s, c in zip(batch["original_sizes"], batch["crop_top_lefts_list"]):
                time_ids.append(compute_time_ids(s, c))
            add_time_ids = torch.cat(time_ids)

            # concat embeddings
            vector_embedding = torch.cat([pool2, embs], dim=1).to(
                device=accelerator.device, dtype=self.weight_dtype
            )
            text_embedding = torch.cat(
                [encoder_hidden_state1, encoder_hidden_state2], dim=2
            ).to(device=accelerator.device, dtype=self.weight_dtype)

            # Sample noise, sample a random timestep for each image, and add noise to the latents,
            # with noise offset and/or multires noise if specified
            noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
                self.args,
                noise_scheduler,
                latent,
                epoch,
                step,
                self.weight_dtype,
            )

            unet_added_conditions = {
                "time_ids": add_time_ids,
                "text_embeds": pool2,
            }

            with accelerator.autocast():
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    text_embedding,
                    added_cond_kwargs=unet_added_conditions,
                ).sample

        return noise_pred, noise, timesteps

    def save_text_encoder(self, model_type, path):
        if model_type == 1:
            text_encoder_bak = CLIPTextModel.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="text_encoder"
            ).to("npu")
            text_encoder_bak.load_state_dict(
                self.text_encoder1.state_dict(), strict=False
            )
        else:
            text_encoder_bak = CLIPTextModelWithProjection.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="text_encoder_2"
            ).to("npu")
            text_encoder_bak.load_state_dict(
                self.text_encoder2.state_dict(), strict=False
            )
        text_encoder_bak.save_pretrained(path)

    def save_pretrained(self, path):
        self.unet.save_pretrained(os.path.join(path, "unet"))
        self.save_text_encoder(1, os.path.join(path, "text_encoder"))
        self.save_text_encoder(2, os.path.join(path, "text_encoder_2"))
