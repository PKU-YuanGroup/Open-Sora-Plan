import torch
import torch.nn.functional as F
try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
import random
from enum import Enum, auto

from einops import rearrange
import cv2

class MaskType(Enum):
    t2iv = auto()
    i2v = auto()
    transition = auto()
    v2v = auto()
    clear = auto()
    random_temporal = auto()
    pos_mask = auto()

TYPE_TO_STR = {
    MaskType.t2iv: 't2iv',
    MaskType.i2v: 'i2v',
    MaskType.transition: 'transition',
    MaskType.v2v: 'v2v',
    MaskType.clear: 'clear',
    MaskType.random_temporal: 'random_temporal',
    MaskType.pos_mask : 'pos_mask'
}

STR_TO_TYPE = {v: k for k, v in TYPE_TO_STR.items()}

def read_video_to_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    cap.release()
    
    # 转换为tensor并调整形状
    tensor = torch.tensor(frames).unsqueeze(1)  # 添加通道维度
    tensor = tensor.permute(0, 1, 2, 3)  # 变换维度为 (T, 1, H, W)


    return tensor

class MaskProcessor:
    def __init__(self, min_clear_ratio=0.0, max_clear_ratio=1.0, resize_transform, **kwargs):
        self.min_clear_ratio = min_clear_ratio
        self.max_clear_ratio = max_clear_ratio
        self.resize_transform = resize_transform
        assert self.min_clear_ratio >= 0 and self.min_clear_ratio <= 1, 'min_clear_ratio should be in the range of [0, 1].'
        assert self.max_clear_ratio >= 0 and self.max_clear_ratio <= 1, 'max_clear_ratio should be in the range of [0, 1].'
        assert self.min_clear_ratio < self.max_clear_ratio, 'min_clear_ratio should be less than max_clear_ratio.'
        self.init_mask_func()

    def t2iv(self, mask):
        mask[:] = 1
        return mask

    def i2v(self, mask):
        mask[0] = 0
        return mask

    def transition(self, mask):
        mask[0] = 0
        mask[-1] = 0
        return mask
    
    def v2v(self, mask):
        num_frames = mask.shape[0]
        end_idx = random.randint(int(num_frames * self.min_clear_ratio), int(num_frames * self.max_clear_ratio))
        mask[:end_idx] = 0
        return mask
    
    def clear(self, mask):
        mask[:] = 0
        return mask
    
    def random_temporal(self, mask):
        num_frames = mask.shape[0]
        num_to_select = random.randint(int(num_frames * self.min_clear_ratio), int(num_frames * self.max_clear_ratio))
        selected_indices = random.sample(range(num_frames), num_to_select)
        mask[selected_indices] = 0
        return mask
    
    def pos_mask(self, maskpath):
        mask = read_video_to_tensor(maskpath)

        mask = self.resize_transform(mask)
        return mask

    def init_mask_func(self):
        self.mask_functions = {
            MaskType.t2iv: self.t2iv,
            MaskType.i2v: self.i2v,
            MaskType.transition: self.transition,
            MaskType.v2v: self.v2v,
            MaskType.clear: self.clear,
            MaskType.random_temporal: self.random_temporal,
            MaskType.pos_mask : self.pos_mask
        }
    
    def __call__(self, pixel_values, maskpath, mask_type_ratio_dict={MaskType.random_temporal: 1.0}):
        T, C, H, W = pixel_values.shape
        mask = torch.ones([T, 1, H, W], device=pixel_values.device, dtype=pixel_values.dtype)
        assert isinstance(mask_type_ratio_dict, dict), 'mask_type_ratio_dict should be a dict.'
        assert mask_type_ratio_dict.keys() <= set(MaskType), f'Invalid mask type: {set(MaskType) - mask_type_ratio_dict.keys()}'

        mask_func_name = random.choices(list(mask_type_ratio_dict.keys()), list(mask_type_ratio_dict.values()))[0]
        if mask_func_name == MaskType.pos_mask:
            mask = self.mask_functions[mask_func_name](maskpath)
        else:
            mask = self.mask_functions[mask_func_name](mask)

        masked_pixel_values = pixel_values * (mask < 0.5)
        return dict(mask=mask, masked_pixel_values=masked_pixel_values)
    
class MaskCompressor:
    def __init__(self, ae_stride_h=8, ae_stride_w=8, ae_stride_t=4, **kwargs):
        self.ae_stride_h = ae_stride_h
        self.ae_stride_w = ae_stride_w
        self.ae_stride_t = ae_stride_t
    
    def compress_mask(self, mask):
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
        mask = mask.transpose(1, 2).contiguous()
        return mask
    
    def __call__(self, mask):
        return self.compress_mask(mask)