from typing import List

from .score import Score

from .constants import HF_CACHE_DIR

from .models.vqascore_models import list_all_vqascore_models, get_vqascore_model

class VQAScore(Score):
    def prepare_scoremodel(self,
                           model='clip-flant5-xxl',
                           device='cuda',
                           cache_dir=HF_CACHE_DIR,
                           **kwargs):
        return get_vqascore_model(
            model,
            device=device,
            cache_dir=cache_dir,
            **kwargs
        )
            
    def list_all_models(self) -> List[str]:
        return list_all_vqascore_models()