from .dit.modeling_dit import DiT_models
from .latte.modeling_latte import Latte_models
# NOTE: vanilla model definition, which only supports DDP training.
# from .dit.dit import DiT_models
# from .latte.latte import Latte_models
from .latte_t2v.latte_t2v import LatteT2V

Diffusion_models = {}
Diffusion_models.update(DiT_models)
Diffusion_models.update(Latte_models)

def get_t2v_models(args):
    return LatteT2V.from_pretrained_2d(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length)
    