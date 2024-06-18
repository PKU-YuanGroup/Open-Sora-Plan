import torch

from safetensors.torch import load_file as safe_load

path = "/storage/ongoing/new/Open-Sora-Plan/bs32_1node_480p_lr1e-4_snr5_noioff0.02_ema_uditultra22_ds22_mt5xxl/checkpoint-500/model/diffusion_pytorch_model.safetensors"
ckpt = safe_load(path, device="cpu")
new_ckpt = {}
k_size = 3
t_stride = 1
for k, v in ckpt.items():
    if 'pos_embed.proj.weight' in k:
        new_v = v.unsqueeze(-3).repeat(1, 1, k_size, 1, 1)  # 768, 4, 3, 3 -> 768, 4, 3, 3, 3
    elif 'attn1.downsampler.layer.weight' in k:
        new_v = v.unsqueeze(-3).repeat(1, 1, k_size, 1, 1)  # 768, 4, 3, 3 -> 768, 4, 3, 3, 3
    elif 'body.0.weight' in k and 'down' in k:
        in_c = v.shape[0]
        new_v = v[:in_c//2].unsqueeze(-3).repeat(1, 1, k_size, 1, 1)  # 384, 768, 3, 3 -> 192, 768, 3, 3, 3
    elif 'body.0.weight' in k and 'up' in k:
        new_v = v.unsqueeze(-3).repeat(2, 1, k_size, 1, 1)  # 6144, 3072, 3, 3 -> 12288, 3072, 3, 3, 3
    elif 'proj_out' in k:
        if 'weight' in k:
            new_v = v.repeat(t_stride, 1)  # 16, 768 -> 32, 768
        elif 'bias' in k:
            new_v = v.repeat(t_stride)  # 16 -> 32
    else:
        new_v = v
    new_ckpt[k] = new_v
torch.save(new_ckpt, 'convert_weight.pt')