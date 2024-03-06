import requests
from tqdm import tqdm
import os
import gdown
import torch

from .vqvae import VQVAE
from .gpt import VideoGPT


def download(id, fname, root=os.path.expanduser('~/.cache/videogpt')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    gdown.download(id=id, output=destination, quiet=False)
    return destination


_VQVAE = {
    'bair_stride4x2x2': '1iIAYJ2Qqrx5Q94s5eIXQYJgAydzvT_8L', # trained on 16 frames of 64 x 64 images
    'ucf101_stride4x4x4': '1uuB_8WzHP_bbBmfuaIV7PK_Itl3DyHY5', # trained on 16 frames of 128 x 128 images
    'kinetics_stride4x4x4': '1DOvOZnFAIQmux6hG7pN_HkyJZy3lXbCB', # trained on 16 frames of 128 x 128 images
    'kinetics_stride2x4x4': '1jvtjjtrtE4cy6pl7DK_zWFEPY3RZt2pB' # trained on 16 frames of 128 x 128 images
}

def load_vqvae(model_name, device=torch.device('cpu'), root=os.path.expanduser('~/.cache/videogpt')):
    assert model_name in _VQVAE, f"Invalid model_name: {model_name}"
    filepath = download(_VQVAE[model_name], model_name, root=root)
    vqvae = VQVAE.load_from_checkpoint(filepath).to(device)
    vqvae.eval()

    return vqvae


_VIDEOGPT = {
    'bair_gpt': '1fNTtJAgO6grEtPNrufkpbee1CfGztW-1', # 1-frame conditional, 16 frames of 64 x 64 images
    'ucf101_uncond_gpt': '1QkF_Sb2XVRgSbFT_SxQ6aZUeDFoliPQq', # unconditional, 16 frames of 128 x 128 images
}

def load_videogpt(model_name, device=torch.device('cpu')):
    assert model_name in _VIDEOGPT, f"Invalid model_name: {model_name}"
    filepath = download(_VIDEOGPT[model_name], model_name)
    gpt = VideoGPT.load_from_checkpoint(filepath).to(device)
    gpt.eval()

    return gpt


_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'

def load_i3d_pretrained(device=torch.device('cpu')):
    from .fvd.pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d
