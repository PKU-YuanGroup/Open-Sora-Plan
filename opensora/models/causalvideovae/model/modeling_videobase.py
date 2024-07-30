import torch
from diffusers import ModelMixin, ConfigMixin
from torch import nn
import os
import json
import pytorch_lightning as pl
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from typing import Optional, Union
import glob


class VideoBaseAE(ModelMixin, ConfigMixin):
    config_name = "config.json"
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def encode(self, x: torch.Tensor, *args, **kwargs):
        pass

    def decode(self, encoding: torch.Tensor, *args, **kwargs):
        pass
    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps
    
        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     
    
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)
    
        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        ckpt_files = glob.glob(os.path.join(pretrained_model_name_or_path, '*.ckpt'))
        if ckpt_files:
            # Adapt to checkpoint
            last_ckpt_file = ckpt_files[-1]
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            model = cls.from_config(config_file)
            model.init_from_ckpt(last_ckpt_file)
            return model
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)