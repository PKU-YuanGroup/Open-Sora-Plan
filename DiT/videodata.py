import math
import os

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torch.nn import functional as F
import random



class LongSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert len(x.shape) == 4
        assert x.dtype == torch.float32
        c, t, h, w = x.shape
        if w < h:
            new_w = int(math.floor((float(w) / h) * self._size))
            new_h = self._size
        else:
            new_w = self._size
            new_h = int(math.floor((float(h) / w) * self._size))
        return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=self._interpolation, align_corners=False)

class UCF101ClassConditionedDataset(Dataset):
    def __init__(self, root_dir, sample_rate, num_frames, max_image_size, dynamic_frames=False):
        self.root_dir = root_dir

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.sample_frames_len = self.sample_rate * self.num_frames
        self.max_image_size = max_image_size
        self.dynamic_frames = dynamic_frames
        self.transform = Compose(
            [
                Lambda(lambda x: ((x / 255.0) - 0.5)),
                LongSideScale(size=self.max_image_size),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    def _make_dataset(self):
        dataset = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.endswith('.avi'):
                    item = (os.path.join(class_path, fname), self.class_to_idx[class_name])
                    dataset.append(item)
        return dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        try:
            video_data = self.read_video(video_path)
            video_outputs = self.transform(video_data)
            # video_outputs = torch.rand(3, 16, 128, 128)
            return video_outputs, label
        except Exception as e:
            print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, self.__len__()-1))


    def read_video(self, video_path):
        decord_vr = VideoReader(video_path, ctx=cpu(0))

        total_frames = len(decord_vr)
        if total_frames > self.sample_frames_len:
            s = random.randint(0, total_frames - self.sample_frames_len - 1)
            e = s + self.sample_frames_len
            num_frames = self.num_frames
        else:
            s = 0
            e = total_frames
            num_frames = int(total_frames / self.sample_frames_len * self.num_frames)
            print(f'sample_frames_len {self.sample_frames_len}, only can {num_frames*self.sample_rate}', video_path, total_frames)

        # random drop to dynamic input frames
        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        if self.dynamic_frames and total_frames > self.sample_frames_len:  # actually only second-half is dynamic, because num_frames are rare...
            cut_idx = random.randint(num_frames // 2, num_frames)
            frame_id_list = frame_id_list[:cut_idx]

        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        return video_data

def pad_to_multiple(number, ds_stride):
    remainder = number % ds_stride
    if remainder == 0:
        return number
    else:
        padding = ds_stride - remainder
        return number + padding

class Collate:
    def __init__(self, max_image_size, vae_stride, patch_size, patch_size_t, num_frames):
        self.max_image_size = max_image_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.num_frames = num_frames

    def __call__(self, batch):
        batch_tubes, labels = tuple(zip(*batch))
        labels = torch.as_tensor(labels).to(torch.long)
        ds_stride = self.vae_stride * self.patch_size
        t_ds_stride = self.vae_stride * self.patch_size_t

        # pad to max multiple of ds_stride
        batch_input_size = [i.shape for i in batch_tubes]
        max_t, max_h, max_w = max([i[1] for i in batch_input_size]), \
                              max([i[2] for i in batch_input_size]), \
                              max([i[3] for i in batch_input_size])
        pad_max_t, pad_max_h, pad_max_w = pad_to_multiple(max_t, t_ds_stride), \
                                          pad_to_multiple(max_h, ds_stride), \
                                          pad_to_multiple(max_w, ds_stride)
        each_pad_t_h_w = [[pad_max_t - i.shape[1],
                           pad_max_h - i.shape[2],
                           pad_max_w - i.shape[3]] for i in batch_tubes]
        pad_batch_tubes = [F.pad(im,
                                 (0, pad_w,
                                  0, pad_h,
                                  0, pad_t), value=0) for (pad_t, pad_h, pad_w), im in zip(each_pad_t_h_w, batch_tubes)]
        pad_batch_tubes = torch.stack(pad_batch_tubes, dim=0)

        # make attention_mask
        max_tube_size = [pad_max_t, pad_max_h, pad_max_w]
        max_latent_size = [max_tube_size[0] // self.vae_stride,
                           max_tube_size[1] // self.vae_stride,
                           max_tube_size[2] // self.vae_stride]
        max_patchify_latent_size = [max_latent_size[0] // self.patch_size_t,
                                    max_latent_size[1] // self.patch_size,
                                    max_latent_size[2] // self.patch_size]
        valid_patchify_latent_size = [[int(math.ceil(i[1] / t_ds_stride)),
                                       int(math.ceil(i[2] / ds_stride)),
                                       int(math.ceil(i[3] / ds_stride))] for i in batch_input_size]
        attention_mask = [F.pad(torch.ones(i),
                                (0, max_patchify_latent_size[2] - i[2],
                                 0, max_patchify_latent_size[1] - i[1],
                                 0, max_patchify_latent_size[0] - i[0]), value=0) for i in valid_patchify_latent_size]
        attention_mask = torch.stack(attention_mask)

        return pad_batch_tubes, labels, attention_mask


