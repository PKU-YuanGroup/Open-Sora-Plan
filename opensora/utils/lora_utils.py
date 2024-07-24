
from peft import get_peft_model, PeftModel
import os
from copy import deepcopy
import torch
from diffusers.training_utils import EMAModel

class EMAModel_LoRA(EMAModel):
    def __init__(self, lora_config, **kwargs):
        super().__init__(**kwargs)
        self.lora_config = lora_config
    
    @classmethod
    def from_pretrained(cls, path, model_cls, lora_config, origin_model_path) -> "EMAModel":
        # 1. load origin model
        origin_model = model_cls.from_pretrained(origin_model_path)  # origin_model
        # 2. convert to lora model automatically and load lora weight
        lora_model = PeftModel.from_pretrained(origin_model, path)  # lora_origin_model
        # 3. ema the whole lora_model
        ema_model = cls(lora_config, parameters=lora_model.parameters(), model_cls=model_cls, model_config=origin_model.config)
        # 4. load ema_kwargs, e.g decay...
        ema_kwargs = torch.load(os.path.join(path, 'ema_kwargs.pt'))
        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")
        # 1. init a base model randomly
        model = self.model_cls.from_config(self.model_config)
        # 2. convert lora_model
        lora_model = get_peft_model(model, self.lora_config)
        # 3. ema_lora_model weight to lora_model
        self.copy_to(lora_model.parameters())
        # 4. save lora weight
        lora_model.save_pretrained(path)  # only lora weight
        # 5. save ema_kwargs, e.g decay...
        state_dict = self.state_dict()  # lora_model weight
        state_dict.pop("shadow_params", None)
        torch.save(state_dict, os.path.join(path, 'ema_kwargs.pt'))


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return
