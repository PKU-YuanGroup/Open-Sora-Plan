import torch

from transformers.generation.streamers import TextStreamer
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.generation_mixin import GenerationMixin
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS


class LlavaPipeline(GenerationMixin, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, tokenizer, language_model, image_processor, args):

        self.tokenizer = tokenizer
        self.vlmodel = language_model
        self.model = language_model.text_decoder
        self.image_processor = image_processor
        self.args = args
        self.config = self.vlmodel.config
        self.generation_config = args.mm.model.generation_config

        self.device = args.mm.model.device
        self.main_input_name = 'input_ids'
        self.model.to(self.device)

    def __call__(self, prompt, image, device, dtype=torch.float16):

        prompt = self.format_prompt(prompt, mm_use_im_start_end=False)
        system = "A chat between a curious human and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the human's questions."
        roles = [["Human", prompt], ["Assistant", None]]
        prompt = self.prompt_template(system, roles, sep="###")

        image_size = image[0].size
        image_tensor = self.process_images(image, image_aspect_ratio="pad")
        if isinstance(image_tensor, list):
            image_tensor = [image.to(device, dtype=dtype) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=dtype)
        input_ids = self.tokenizer_image_token(prompt, MODEL_CONSTANTS["llava"]["IMAGE_TOKEN_INDEX"],
                                               return_tensors='pt').unsqueeze(0).to(device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        attention_mask = torch.ones(input_ids.shape).bool().to(device)
        (inputs,
         position_ids,
         attention_mask,
         _,
         inputs_embeds,
         _) = self.vlmodel.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            images=image_tensor,
            image_sizes=image_size
        )

        causal_attention_mask = torch.triu(
            torch.ones(inputs_embeds.shape[0], 1, inputs_embeds.shape[1], inputs_embeds.shape[1],
                       device=inputs_embeds.device),
            diagonal=1
        ).bool()
        attention_mask = ~attention_mask
        expanded_attention_mask = attention_mask[:, None, None, :].expand(
            inputs_embeds.shape[0], 1, inputs_embeds.shape[1], inputs_embeds.shape[1]
        )
        attention_mask = causal_attention_mask.masked_fill(expanded_attention_mask, True)

        inputs_embeds = inputs_embeds.transpose(0, 1)
        self.generate(position_ids=position_ids,
                      attention_mask=attention_mask,
                      inputs_embeds=inputs_embeds,
                      do_sample=True if self.generation_config.temperature > 0 else False,
                      temperature=self.generation_config.temperature,
                      max_new_tokens=self.generation_config.max_new_tokens,
                      streamer=streamer,
                      use_cache=True)

    def tokenizer_image_token(self, prompt, image_token_index=MODEL_CONSTANTS["llava"]["IMAGE_TOKEN_INDEX"],
                              return_tensors="pt"):
        prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in
                         prompt.split(MODEL_CONSTANTS["llava"]["IMAGE_TOKEN"])]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    @staticmethod
    def format_prompt(prompt, mm_use_im_start_end):
        if mm_use_im_start_end:
            prompt = MODEL_CONSTANTS["llava"]["IMG_START_TOKEN"] + MODEL_CONSTANTS["llava"]["IMAGE_TOKEN"] + \
                     MODEL_CONSTANTS["llava"]["IMG_END_TOKEN"] + '\n' + prompt
        else:
            prompt = MODEL_CONSTANTS["llava"]["IMAGE_TOKEN"] + '\n' + prompt
        return prompt

    def prompt_template(self, system: str, roles_prompts: list, sep: str):

        ret = system + sep
        for role, message in roles_prompts:
            if message:
                ret += role + ": " + message + sep
            else:
                ret += role + ":"
        return ret

    def _prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"decoder_input": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {"input_ids": None,
             "position_ids": position_ids,
             "attention_mask": attention_mask,
             }
        )
        return model_inputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self._prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs