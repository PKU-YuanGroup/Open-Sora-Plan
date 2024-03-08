from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"
model_path = "liuhaotian/llava-v1.6-34b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    cache_dir='./cache_dir'
)

