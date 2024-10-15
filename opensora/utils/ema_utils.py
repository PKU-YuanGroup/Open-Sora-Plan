
from peft import get_peft_model, PeftModel
import os
from copy import deepcopy
import torch
import json
from diffusers.training_utils import EMAModel as diffuser_EMAModel



class EMAModel(diffuser_EMAModel):
    def __init__(self, parameters, **kwargs):
        self.lora_config = kwargs.pop('lora_config', None)
        super().__init__(parameters, **kwargs)
    
    @classmethod
    def from_pretrained(cls, path, model_cls, lora_config, model_base) -> "EMAModel":
        # 1. load model
        if lora_config is not None:
            # 1.1 load origin model
            model_base = model_cls.from_pretrained(model_base)  # model_base
            config = model_base.config
            # 1.2 convert to lora model automatically and load lora weight
            model = PeftModel.from_pretrained(model_base, path)  # lora_origin_model
        else:
            model = model_cls.from_pretrained(path)
            config = model.config
        # 3. ema the whole model
        ema_model = cls(model.parameters(), model_cls=model_cls, model_config=config, lora_config=lora_config)
        # 4. load ema_config, e.g decay...
        with open(os.path.join(path, 'ema_config.json'), 'r') as f:
            state_dict = json.load(f)
        ema_model.load_state_dict(state_dict)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")
        # 1. init a base model randomly
        model = self.model_cls.from_config(self.model_config)
        # 1.1 convert lora_model
        if self.lora_config is not None:
            model = get_peft_model(model, self.lora_config)
        # 2. ema_model copy to model
        self.copy_to(model.parameters())
        # 3. save weight
        if self.lora_config is not None:
            model.save_pretrained(path)  # only lora weight
            merge_model = model.merge_and_unload()
            merge_model.save_pretrained(path) # merge_model weight
        else:
            merge_model.save_pretrained(path) # model weight
        # 4. save ema_config, e.g decay...
        state_dict = self.state_dict()  # lora_model weight
        state_dict.pop("shadow_params", None)
        with open(os.path.join(path, 'ema_config.json'), 'w') as f:
            json.dump(state_dict, f, indent=2)