import torch

from safetensors.torch import load_file as safe_load

path = "diffusion_pytorch_model.safetensors"
ckpt = safe_load(path, device="cpu")
new_ckpt = {}
for k, v in ckpt.items():
    if 'transformer_blocks' in k:
        split = k.split('.')
        idx = int(split[1])
        if idx % 2 == 0:
            split[1] = str(idx//2)
        else:
            split[0] = 'temporal_transformer_blocks'
            split[1] = str((idx-1)//2)
        new_k = '.'.join(split)
    else:
        new_k = k
    new_ckpt[new_k] = v
torch.save(new_ckpt, 'convert_weight.pt')