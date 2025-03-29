
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constants import HF_CACHE_DIR
from .vqascore import VQAScore, list_all_vqascore_models

def list_all_models():
    return list_all_vqascore_models()

def get_score_model(model='clip-flant5-xxl', device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    return VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)