from typing import List
import torch

from .vqa_model import VQAScoreModel
from .mm_utils import expand2square, load_pretrained_model, t5_tokenizer_image_token
from ...constants import HF_CACHE_DIR, CONTEXT_LEN, SYSTEM_MSG, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from .clip_t5.model import CLIPT5ForConditionalGeneration, ModelArguments

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

def format_question(question, conversation_style='plain'):
    if conversation_style == 't5_plain': # for 1st stage t5 model
        question = DEFAULT_IMAGE_TOKEN + question
    elif conversation_style == 't5_chat': # for 2nd stage t5 model
        question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system': # for 2nd stage t5 model
        question = "USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system_no_user': # for 2nd stage t5 model
        question = "" + DEFAULT_IMAGE_TOKEN + "\n" + question + " : "
    # elif conversation_style == 't5_chat_ood_system': # for 2nd stage t5 model
    #     question = SYSTEM_MSG + " HUMAN: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " GPT: "
    else:
        raise NotImplementedError()
    return question

def format_answer(answer, conversation_style='plain'):
    return answer

CLIP_T5_MODELS = {
    # We recommend using 'clip-flant5-xxl' for maximal performance.
    # If you want to use a smaller model, we recommend using 'clip-flant5-xl'.
    'clip-flant5-xxl': {
        'tokenizer' : {
            'path': 'google/flan-t5-xxl',
            'model_max_length': CONTEXT_LEN,
        },
        'model': {
            'path': 'zhiqiulin/clip-flant5-xxl',
            'conversation': 't5_chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'clip-flant5-xl': {
        'tokenizer' : {
            'path': 'google/flan-t5-xl',
            'model_max_length': CONTEXT_LEN,
        },
        'model': {
            'path': 'zhiqiulin/clip-flant5-xl',
            'conversation': 't5_chat',
            'image_aspect_ratio': 'pad',
        },
    },
    # The following models are suboptimal, but are included for completeness.
    # 'clip-flant5-xxl-stage-1': {
    #     'tokenizer' : {
    #         'path': 'google/flan-t5-xxl',
    #         'model_max_length': CONTEXT_LEN,
    #     },
    #     'model': {
    #         'path': 'google/flan-t5-xxl',
    #         'mmprojector_repo': 'zhiqiulin/clip-flant5-xxl-stage-1',
    #         'mmprojector_name': 'mm_projector.bin',
    #         'conversation': "t5_plain",
    #         'image_aspect_ratio': 'square',
    #     },
    # },
    # 'clip-flant5-xxl-no-split-text': {
    #     'tokenizer' : {
    #         'path': 'google/flan-t5-xxl',
    #         'model_max_length': CONTEXT_LEN,
    #     },
    #     'model': {
    #         'path': 'zhiqiulin/clip-flant5-xxl-no-split-text',
    #         'conversation': 't5_chat',
    #         'image_aspect_ratio': 'pad',
    #     },
    # },
    # 'clip-flant5-xxl-stage-1-no-split-text': {
    #     'tokenizer' : {
    #         'path': 'google/flan-t5-xxl',
    #         'model_max_length': CONTEXT_LEN,
    #     },
    #     'model': {
    #         'path': 'google/flan-t5-xxl',
    #         'mmprojector_repo': 'zhiqiulin/clip-flant5-xxl-stage-1-no-split-text',
    #         'mmprojector_name': 'mm_projector.bin',
    #         'conversation': "t5_plain",
    #         'image_aspect_ratio': 'square',
    #     },
    # },
    # 'clip-t5-xxl': {
    #     'tokenizer' : {
    #         'path': 't5-11b',
    #         'model_max_length': CONTEXT_LEN,
    #     },
    #     'model': {
    #         'path': 'zhiqiulin/clip-t5-xxl',
    #         'conversation': 't5_chat',
    #         'image_aspect_ratio': 'pad',
    #     },
    # },
    # 'clip-t5-xxl-stage-1': {
    #     'tokenizer' : {
    #         'path': 't5-11b',
    #         'model_max_length': CONTEXT_LEN,
    #     },
    #     'model': {
    #         'path': 't5-11b',
    #         'mmprojector_repo': 'zhiqiulin/clip-t5-xxl-stage-1',
    #         'mmprojector_name': 'mm_projector.bin',
    #         'conversation': "t5_plain",
    #         'image_aspect_ratio': 'square',
    #     },
    # },
    # 'clip-flant5-xl-stage-1': {
    #     'tokenizer' : {
    #         'path': 'google/flan-t5-xl',
    #         'model_max_length': CONTEXT_LEN,
    #         'padding_side': 'right',
    #     },
    #     'model': {
    #         'path': 'google/flan-t5-xl',
    #         'mmprojector_repo': 'zhiqiulin/clip-flant5-xl-stage-1',
    #         'mmprojector_name': 'mm_projector.bin',
    #         'conversation': "t5_plain",
    #         'image_aspect_ratio': 'square',
    #     },
    # },
    
    ## for prompting ablation
    'clip-flant5-xxl-no-system': {
        'tokenizer' : {
            'path': 'google/flan-t5-xxl',
            'model_max_length': CONTEXT_LEN,
        },
        'model': {
            'path': 'zhiqiulin/clip-flant5-xxl',
            'conversation': 't5_chat_no_system',
            'image_aspect_ratio': 'pad',
        },
    },
    'clip-flant5-xxl-no-system-no-user': {
        'tokenizer' : {
            'path': 'google/flan-t5-xxl',
            'model_max_length': CONTEXT_LEN,
        },
        'model': {
            'path': 'zhiqiulin/clip-flant5-xxl',
            'conversation': 't5_chat_no_system_no_user',
            'image_aspect_ratio': 'pad',
        },
    },
}



class CLIPT5Model(VQAScoreModel):
    """A wrapper for the CLIP-FlanT5 or CLIP-T5 models"""
    def __init__(self,
                 model_name='clip-flant5-xxl',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        self.meta_dict = {
        'tokenizer' : {
            'path': model_name,
            'model_max_length': CONTEXT_LEN,
        },
        'model': {
            'path': model_name,
            'conversation': 't5_chat',
            'image_aspect_ratio': 'pad',
        },
    }
        # assert model_name in CLIP_T5_MODELS
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)
    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        model_args = ModelArguments()
        model_max_length = self.meta_dict['tokenizer']['model_max_length'] \
            if 'model_max_length' in self.meta_dict['tokenizer'] else None
        padding_side = self.meta_dict['tokenizer']['padding_side'] \
            if 'padding_side' in self.meta_dict['tokenizer'] else None
        mmprojector_repo = self.meta_dict['model']['mmprojector_repo'] \
            if 'mmprojector_repo' in self.meta_dict['model'] else None
        mmprojector_name = self.meta_dict['model']['mmprojector_name'] \
            if 'mmprojector_name' in self.meta_dict['model'] else None
        
        # default is 'pad'
        # stage-1 models use 'square'
        self.image_aspect_ratio = self.meta_dict['model']['image_aspect_ratio'] \
            if 'image_aspect_ratio' in self.meta_dict['model'] else 'pad'
        
        self.conversational_style = self.meta_dict['model']['conversation']
        
        self.context_len = CONTEXT_LEN
        
        self.tokenizer, self.model, self.image_processor = load_pretrained_model(
            CLIPT5ForConditionalGeneration,
            model_args,
            model_path=self.meta_dict['model']['path'],
            tokenizer_path=self.meta_dict['tokenizer']['path'],
            model_max_length=model_max_length,
            padding_side=padding_side,
            image_aspect_ratio=self.image_aspect_ratio,
            mmprojector_repo=mmprojector_repo,
            mmprojector_name=mmprojector_name,
            device=self.device,
            cache_dir=self.cache_dir
        )

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        if self.image_aspect_ratio == 'pad':
            image = [expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean)) for image in image]
        image = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image]
        assert all(x.shape == image[0].shape for x in image)
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str=default_question_template,
                answer_template: str=default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        # Formatting for CLIP-FlanT5 desired input including system message and image tokens
        questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]

        images = self.load_images(images)
        
        input_ids = [t5_tokenizer_image_token(qs, self.tokenizer, return_tensors='pt') for qs in questions]
        labels = [t5_tokenizer_image_token(ans, self.tokenizer, return_tensors='pt') for ans in answers]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        decoder_attention_mask = labels.ne(IGNORE_INDEX)
        
        input_ids, attention_mask, decoder_attention_mask, labels = input_ids.to(self.device), \
            attention_mask.to(self.device), decoder_attention_mask.to(self.device), labels.to(self.device)
        model_input_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'images': images,
            'past_key_values': None,
            'inputs_embeds': None,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': True,
        }
        
        outputs = self.model(
            **model_input_kwargs
        )

        logits = outputs.logits
        lm_prob = torch.zeros(logits.shape[0])
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp() # exp to cancel the log and get raw prob between 0 and 1
        return lm_prob
    
    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def generate(self,
                 images: List[str],
                 prompts: List[str],
                 temperature: float=0.2,
                 ):
        """Forward pass of the model to return n strings for n (image, prompt) pairs
        """
        assert len(images) == len(prompts), "Number of images and texts must match"
        
        # Formatting for CLIP-FlanT5 desired input including system message and image tokens
        questions = [format_question(prompt, conversation_style=self.conversational_style) for prompt in prompts]
        images = self.load_images(images)
        
        input_ids = [t5_tokenizer_image_token(qs, self.tokenizer, return_tensors='pt') for qs in questions]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        model_input_kwargs = {
            'inputs': input_ids,
            'images': images,
            'attention_mask': attention_mask,
            "do_sample": True if temperature > 0 else False,
            "temperature": temperature,
            "top_p": None,
            "num_beams": 1,
            "max_new_token": 1024,
            "use_cache": True,
        }
        
        outputs = self.model.generate(
            **model_input_kwargs
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(len(outputs)):
            if outputs[i].endswith(" "):
                outputs[i] = outputs[i][:-1]
            outputs[i] = outputs[i].strip()
        return outputs
