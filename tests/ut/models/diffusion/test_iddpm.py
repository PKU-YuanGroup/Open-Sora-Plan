import torch
from diffusers.schedulers import PNDMScheduler
import mindspeed.megatron_adaptor

from mindspeed_mm.models.diffusion.iddpm import IDDPM
from mindspeed_mm.models.diffusion.diffusers_scheduler import DiffusersScheduler
from tests.ut.utils import judge_expression


COEF1_SUM = torch.Tensor([10.8997])
COEF2_SUM = torch.Tensor([985.4296])

COEF_THRESHOLD = torch.Tensor([0.001])


class TestIDDPM:

    def test_init_from_diffusers(self):
        config = {
            "model_id": "PNDM",
            "num_train_steps": 1000,
            "num_inference_steps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "guidance_scale": 4.5,
        }
        diffusion = DiffusersScheduler(config).diffusion
        judge_expression(isinstance(diffusion, PNDMScheduler))

    def test_sample(self):
        pass

    def test_training_losses(self):
        pass
