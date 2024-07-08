# import torch

# from safetensors.torch import load_file as safe_load

# path = "bs16_4node_480p_lr1e-4_snr5_noioff0.02_ema_rope_uditultra22_qknorm_ds22_mt5xxl_mjencn_czhan_humanimg/checkpoint-73000/model_ema/diffusion_pytorch_model.safetensors"
# ckpt = safe_load(path, device="cpu")
# new_ckpt = {}
# k_size = 3
# patch_size_t = 1
# for k, v in ckpt.items():
#     if 'pos_embed.proj.weight' in k:
#         new_v = v.unsqueeze(-3).repeat(1, 1, k_size, 1, 1) / k_size # 768, 4, 3, 3 -> 768, 4, 3, 3, 3
#     elif 'attn1.downsampler.layer.weight' in k:
#         new_v = v.unsqueeze(-3).repeat(1, 1, k_size, 1, 1) / k_size  # 768, 4, 3, 3 -> 768, 4, 3, 3, 3
#     elif 'body.0.weight' in k and 'down' in k:
#         in_c = v.shape[0]
#         new_v = v[:in_c//2].unsqueeze(-3).repeat(1, 1, k_size, 1, 1)  # 384, 768, 3, 3 -> 192, 768, 3, 3, 3
#     elif 'body.0.weight' in k and 'up' in k:
#         new_v = v.unsqueeze(-3).repeat(2, 1, k_size, 1, 1)  # 6144, 3072, 3, 3 -> 12288, 3072, 3, 3, 3
#     elif 'proj_out' in k:
#         if 'weight' in k:
#             new_v = v.repeat(patch_size_t, 1)  # 16, 768 -> 32, 768
#         elif 'bias' in k:
#             new_v = v.repeat(patch_size_t)  # 16 -> 32
#     else:
#         new_v = v
#     new_ckpt[k] = new_v
# torch.save(new_ckpt, f'480p_73000_ema_k{k_size}_p{patch_size_t}.pt')


import torch

from safetensors.torch import load_file as safe_load


def conv2d_to_conv3d(name, conv2d_weights, k_size):
    conv3d_weights = torch.zeros_like(conv2d_weights).unsqueeze(2).repeat(1, 1, k_size, 1, 1)
    # conv3d_weights[:,:,0,:,:] = conv2d_weights / 10
    conv3d_weights[:,:,1,:,:] = conv2d_weights
    # conv3d_weights[:,:,2,:,:] = conv2d_weights / 10
    return conv3d_weights

path = "bs16_4node_480p_lr1e-4_snr5_noioff0.02_ema_rope_uditultra22_qknorm_ds22_mt5xxl_mjencn_czhan_humanimg/checkpoint-73000/model_ema/diffusion_pytorch_model.safetensors"
ckpt = safe_load(path, device="cpu")
new_ckpt = {}
k_size = 3
patch_size_t = 1
for k, v in ckpt.items():
    if 'pos_embed.proj.weight' in k:
        new_v = conv2d_to_conv3d(k, v, k_size) # 768, 4, 3, 3 -> 768, 4, 3, 3, 3
    elif 'attn1.downsampler.layer.weight' in k:
        new_v = conv2d_to_conv3d(k, v, k_size)  # 768, 4, 3, 3 -> 768, 4, 3, 3, 3
    elif 'body.0.weight' in k and 'down' in k:
        new_v = conv2d_to_conv3d(k, v, k_size)  # 384, 768, 3, 3 -> 384, 768, 3, 3, 3
    elif 'body.0.weight' in k and 'up' in k:
        new_v = conv2d_to_conv3d(k, v, k_size)  # 6144, 3072, 3, 3 -> 6144, 3072, 3, 3, 3
    else:
        new_v = v
    new_ckpt[k] = new_v
torch.save(new_ckpt, f'480p_73000_ema_k{k_size}_p{patch_size_t}_repeat_lowsize10.pt')