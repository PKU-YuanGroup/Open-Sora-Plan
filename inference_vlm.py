import torch
from PIL import Image

import mindspeed.megatron_adaptor
from megatron.training import get_args
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS

from mindspeed_mm.utils.utils import get_dtype
from mindspeed_mm.tasks.inference.pipeline.llava_pipeline import LlavaPipeline
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.models.text_encoder import Tokenizer
from pretrain_llava import model_provider


def load_models(args):
    tokenizer = Tokenizer(args.mm.model.tokenizer).get_tokenizer()
    model = model_provider()
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([MODEL_CONSTANTS["llava"]["IMAGE_PATCH_TOKEN"]], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([MODEL_CONSTANTS["llava"]["IMG_START_TOKEN"], MODEL_CONSTANTS["llava"]["IMG_END_TOKEN"]], special_tokens=True)
    image_processor = model.image_encoder.image_processor
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    model.to(dtype=args.mm.model.dtype, device=torch.device(args.mm.model.device))
    return tokenizer, model, image_processor, context_len


def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return [image]


def main():
    from megatron.training.initialize import initialize_megatron
    from mindspeed_mm.configs.config import merge_mm_args
    initialize_megatron(
        extra_args_provider=mm_extra_args_provider, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
    args = get_args()
    merge_mm_args(args)
    tokenizer, model, image_processor, context_len = load_models(args)
    llava_pipeline = LlavaPipeline(tokenizer, model, image_processor, args)
    image = load_image(args.mm.model.image_path)
    dtype = get_dtype(args.mm.model.dtype)
    llava_pipeline(args.mm.model.prompts, image, args.mm.model.device, dtype)


if __name__ == '__main__':
    main()