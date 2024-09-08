import torch
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append(".")
from opensora.models.ae.videobase import CausalVAEModel, CausalVAEDataset

num_workers = 4
batch_size = 12

torch.manual_seed(0)
torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model_name_or_path = 'results/causalvae/checkpoint-26000'
data_path = '/remote-home1/dataset/UCF-101'
video_num_frames = 17
resolution = 128
sample_rate = 10

vae = CausalVAEModel.load_from_checkpoint(pretrained_model_name_or_path)
vae.to(device)

dataset = CausalVAEDataset(data_path, sequence_length=video_num_frames, resolution=resolution, sample_rate=sample_rate)
subset_indices = list(range(1000))
subset_dataset = Subset(dataset, subset_indices)
loader = DataLoader(subset_dataset, batch_size=8, pin_memory=True)

all_latents = []
for video_data in loader:
    video_data = video_data['video'].to(device)
    latents = vae.encode(video_data).sample()
    all_latents.append(video_data.cpu())

all_latents_tensor = torch.cat(all_latents)
std = all_latents_tensor.std().item()
normalizer = 1 / std
print(f'{normalizer = }')