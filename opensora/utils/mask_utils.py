
from math import floor, ceil
from abc import ABC, abstractmethod
import cv2
import torch
import torch.nn.functional as F
import imageio
import numpy as np
try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
import random
from enum import Enum, auto

from einops import rearrange

class MaskType(Enum):
    t2iv = auto() # For video, execute t2v (all frames are masked), for image, execute t2i (the image are masked)
    i2v = auto() # Only for video, execute i2v (i.e. maintain the first frame and mask the rest)
    transition = auto() # Only for video, execute transition (i.e. maintain the first and last frame and mask the rest)
    continuation = auto() # Only for video, execute video continuation (i.e. maintain the starting k frames and mask the rest)
    clear = auto() # For video and image, all frames are not masked
    random_temporal = auto() # For video, randomly mask some frames

TYPE_TO_STR = {mask_type: mask_type.name for mask_type in MaskType}
STR_TO_TYPE = {mask_type.name: mask_type for mask_type in MaskType}

def save_mask_to_video(mask, save_path='mask.mp4', fps=24):
    T, _, H, W = mask.shape
    writer = imageio.get_writer(save_path, fps=fps, codec='libx264', quality=6)
    for t in range(T):
        frame = mask[t, 0].cpu().numpy() * 255
        frame = frame.astype(np.uint8)  # 确保数据类型是 uint8
        writer.append_data(frame)
    writer.close()

def read_video(video_path):
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)
    video_array = np.stack(frames)
    video_tensor = torch.from_numpy(video_array).float()
    reader.close()
    return video_tensor

class BaseMaskGenerator(ABC):

    def create_system_mask(self, num_frames, height, width, device, dtype):
        if num_frames is None or height is None or width is None:
            raise ValueError('num_frames, height, and width should be provided.')
        return torch.ones([num_frames, 1, height, width], device=device, dtype=dtype)

    @abstractmethod
    def process(self, mask):
        # process self.mask to meet the specific task
        pass

    def __call__(self, num_frames=None, height=None, width=None, device='cuda', dtype=torch.float32):
        mask = self.create_system_mask(num_frames, height, width, device, dtype)
        return self.process(mask)

class T2IVMaskGenerator(BaseMaskGenerator):
    def process(self, mask):
        mask.fill_(1)
        return mask

class I2VMaskGenerator(BaseMaskGenerator):
    def process(self, mask):
        mask[0] = 0
        return mask

class TransitionMaskGenerator(BaseMaskGenerator):
    def process(self, mask):
        mask[0] = 0
        mask[-1] = 0
        return mask

class ContinuationMaskGenerator(BaseMaskGenerator):
    
    def __init__(self, min_clear_ratio=0.0, max_clear_ratio=1.0):
        assert min_clear_ratio >= 0 and min_clear_ratio <= 1, 'min_clear_ratio should be in the range of [0, 1].'
        assert max_clear_ratio >= 0 and max_clear_ratio <= 1, 'max_clear_ratio should be in the range of [0, 1].'
        assert min_clear_ratio <= max_clear_ratio, 'min_clear_ratio should be less than max_clear_ratio.'
        self.min_clear_ratio = min_clear_ratio
        self.max_clear_ratio = max_clear_ratio

    def process(self, mask):
        num_frames = mask.shape[0]
        end_idx = random.randint(floor(num_frames * self.min_clear_ratio), ceil(num_frames * self.max_clear_ratio))
        mask[0:end_idx] = 0
        return mask

class ClearMaskGenerator(BaseMaskGenerator):
    def process(self, mask):
        mask.zero_()
        return mask

class RandomTemporalMaskGenerator(BaseMaskGenerator):

    def __init__(self, min_clear_ratio=0.0, max_clear_ratio=1.0):
        assert min_clear_ratio >= 0 and min_clear_ratio <= 1, 'min_clear_ratio should be in the range of [0, 1].'
        assert max_clear_ratio >= 0 and max_clear_ratio <= 1, 'max_clear_ratio should be in the range of [0, 1].'
        assert min_clear_ratio <= max_clear_ratio, 'min_clear_ratio should be less than max_clear_ratio.'
        self.min_clear_ratio = min_clear_ratio
        self.max_clear_ratio = max_clear_ratio

    def process(self, mask):
        num_frames = mask.shape[0]
        num_to_select = random.randint(floor(num_frames * self.min_clear_ratio), ceil(num_frames * self.max_clear_ratio))
        selected_indices = random.sample(range(num_frames), num_to_select)
        mask[selected_indices] = 0
        return mask


class MaskProcessor:
    def __init__(
        self, 
        max_height=640, 
        max_width=640, 
        min_clear_ratio=0.0, 
        max_clear_ratio=1.0, 
    ):
        
        self.max_height = max_height
        self.max_width = max_width
        self.min_clear_ratio = min_clear_ratio
        self.max_clear_ratio = max_clear_ratio

        self.init_mask_generators()

    def init_mask_generators(self):
        self.mask_generators = {
            MaskType.t2iv: T2IVMaskGenerator(),
            MaskType.i2v: I2VMaskGenerator(),
            MaskType.transition: TransitionMaskGenerator(),
            MaskType.continuation: ContinuationMaskGenerator(min_clear_ratio=self.min_clear_ratio, max_clear_ratio=self.max_clear_ratio),
            MaskType.clear: ClearMaskGenerator(),
            MaskType.random_temporal: RandomTemporalMaskGenerator(min_clear_ratio=self.min_clear_ratio, max_clear_ratio=self.max_clear_ratio),
        }
    
    def get_mask(self, mask_generator_type, num_frames, height, width, device='cuda', dtype=torch.float32):
        return self.mask_generators[mask_generator_type](num_frames, height, width, device=device, dtype=dtype)
    
    def __call__(self, pixel_values, mask_type=None, mask_type_ratio_dict=None):

        num_frames, _, height, width = pixel_values.shape   

        if mask_type_ratio_dict is not None:  
            assert isinstance(mask_type_ratio_dict, dict), 'mask_type_ratio_dict should be a dict.'
            assert mask_type_ratio_dict.keys() <= set(MaskType), f'Invalid mask type: {set(MaskType) - mask_type_ratio_dict.keys()}'
            mask_generator_type = random.choices(list(mask_type_ratio_dict.keys()), list(mask_type_ratio_dict.values()))[0]
        elif mask_type is not None:
            assert mask_type in STR_TO_TYPE.keys() or mask_type in STR_TO_TYPE.values(), f'Invalid mask type: {mask_type}'
            mask_generator_type = mask_type if mask_type in MaskType else STR_TO_TYPE[mask_type]
        else:
            raise ValueError('mask_type or mask_type_ratio_dict should be provided.')
        
        mask = self.get_mask(mask_generator_type, num_frames, height, width, device=pixel_values.device, dtype=pixel_values.dtype)

        masked_pixel_values = pixel_values * (mask < 0.5)
        return dict(mask=mask, masked_pixel_values=masked_pixel_values)
    
class MaskCompressor:
    def __init__(self, ae_stride_h=8, ae_stride_w=8, ae_stride_t=4, **kwargs):
        self.ae_stride_h = ae_stride_h
        self.ae_stride_w = ae_stride_w
        self.ae_stride_t = ae_stride_t
    
    def __call__(self, mask):
        B, C, T, H, W = mask.shape
        new_H, new_W = H // self.ae_stride_h, W // self.ae_stride_w
        mask = rearrange(mask, 'b c t h w -> (b c t) 1 h w')
        if torch_npu is not None:
            dtype = mask.dtype
            mask = mask.to(dtype=torch.float32)
            mask = F.interpolate(mask, size=(new_H, new_W), mode='bilinear')
            mask = mask.to(dtype)
        else:
            mask = F.interpolate(mask, size=(new_H, new_W), mode='bilinear')
        mask = rearrange(mask, '(b c t) 1 h w -> b c t h w', t=T, b=B)
        if T % 2 == 1:
            new_T = T // self.ae_stride_t + 1
            mask_first_frame = mask[:, :, 0:1].repeat(1, 1, self.ae_stride_t, 1, 1).contiguous() 
            mask = torch.cat([mask_first_frame, mask[:, :, 1:]], dim=2)
        else:
            new_T = T // self.ae_stride_t
        mask = mask.view(B, new_T, self.ae_stride_t, new_H, new_W)
        mask = mask.transpose(1, 2).contiguous() # Transpose to allows the channel dimension to represent a portion of the region in the original mask
        return mask
    
class BaseNoiseAdder(ABC):
    
    @abstractmethod
    def add_noise(self, mask_pixel_values, mask):
        pass

    def __call__(self, mask_pixel_values, mask):
        return self.add_noise(mask_pixel_values, mask)
    
class GaussianNoiseAdder(BaseNoiseAdder):
    def __init__(self, mean=-3.0, std=0.5, clear_ratio=0.05):
        self.mean = mean
        self.std = std
        self.clear_ratio = clear_ratio
    # pixel_values: (B, C, T, H, W)
    # mask: (B, 1, T, H, W)
    def add_noise(self, masked_pixel_values, mask):
        if random.random() < self.clear_ratio:
            return masked_pixel_values
        noise_sigma = torch.normal(mean=self.mean, std=self.std, size=(masked_pixel_values.shape[0],), device=masked_pixel_values.device)
        noise_sigma = torch.exp(noise_sigma).to(dtype=masked_pixel_values.dtype)
        noise = torch.randn_like(masked_pixel_values) * noise_sigma[:, None, None, None, None]
        noise = torch.where(mask < 0.5, noise, torch.zeros_like(noise))
        return masked_pixel_values + noise


if __name__ == '__main__':
    video_path = '/home/image_data/hxy/data/video/000184.mp4'
    video = read_video(video_path)
    processor = MaskProcessor()
    ratio_dict = {
        MaskType.t2iv: 0,
        MaskType.i2v: 0,
        MaskType.transition: 0,
        MaskType.continuation: 0,
        MaskType.clear: 0,
        MaskType.random_temporal: 1,
    }

    mask = processor(video, mask_type_ratio_dict=ratio_dict)['mask']
    print(mask.shape)
    save_mask_to_video(mask, save_path='test_mask.mp4', fps=24)
    