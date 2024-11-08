from mindspeed_mm.tasks.inference.pipeline.opensora_pipeline import OpenSoraPipeline
from mindspeed_mm.tasks.inference.pipeline.opensoraplan_pipeline import OpenSoraPlanPipeline
from mindspeed_mm.tasks.inference.pipeline.cogvideox_pipeline import CogVideoXPipeline

SoraPipeline_dict = {"OpenSoraPlanPipeline": OpenSoraPlanPipeline,
                     "OpenSoraPipeline": OpenSoraPipeline,
                     "CogVideoXPipeline": CogVideoXPipeline}

__all__ = ["SoraPipeline_dict"]
