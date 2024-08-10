import torch
from diffusers.schedulers import PNDMScheduler

from mindspeed_mm.models.diffusion.iddpm import IDDPM
from mindspeed_mm.models.diffusion.diffusers_scheduler import DiffusersScheduler
from tests.ut.utils import judge_expression


COEF1_SUM = torch.Tensor([10.8997])
COEF2_SUM = torch.Tensor([985.4296])

COEF_THRESHOLD = torch.Tensor([0.001])


class TestIDDPM:
    def test_init(self):
        diffusion = IDDPM()
        coef1_sum = diffusion.posterior_mean_coef1.sum().cpu()
        coef2_sum = diffusion.posterior_mean_coef2.sum().cpu()
        judge_expression(abs(coef1_sum - COEF1_SUM) < COEF_THRESHOLD)
        judge_expression(abs(coef2_sum - COEF2_SUM) < COEF_THRESHOLD)

    def test_init_from_diffusers(self):
        config = {
            "model_id": "PNDM",
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "guidance_scale": 4.5,
            "num_timesteps": 1000,
            "device": "npu"
        }
        diffusion = DiffusersScheduler(config).diffusion
        judge_expression(isinstance(diffusion, PNDMScheduler))

    def test_sample(self):
        pass

    def test_training_losses(self):
        pass

