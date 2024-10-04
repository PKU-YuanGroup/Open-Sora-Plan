
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
    # temporal-level mask
    t2iv = auto() # For video, execute t2v (all frames are masked), for image, execute t2i (the image are masked)
    i2v = auto() # Only for video, execute i2v (i.e. maintain the first frame and mask the rest)
    transition = auto() # Only for video, execute transition (i.e. maintain the first and last frame and mask the rest)
    continuation = auto() # Only for video, execute video continuation (i.e. maintain the starting k frames and mask the rest)
    clear = auto() # For video and image, all frames are not masked
    random_temporal = auto() # For video, randomly mask some frames
    # spatial-level mask
    semantic = auto() # For video and image, mask a semantic region (such as an object)
    dilate = auto() # For video and image, dilate the semantic mask to include more pixels and improve generalization ability
    bbox = auto() # For video and image, mask a bounding box region (such as a bbox of an object)
    random_spatial = auto() # For video and image, randomly mask a spatial region
    # outpainting mask
    outpaint_random_spatial = auto() # For video and image, reverse the random_spatial type mask and use it as the outpainting
    outpaint_semantic = auto() # For video and image, reverse the semantic type mask and use it as the outpainting
    outpaint_bbox = auto() # For video and image, reverse the bbox type mask and use it as the outpainting
    # mixed mask
    i2v_outpaint_semantic = auto() # For video, a semantic region of the first frame is not masked, and the rest are masked

TYPE_TO_STR = {mask_type: mask_type.name for mask_type in MaskType}
STR_TO_TYPE = {mask_type.name: mask_type for mask_type in MaskType}

def save_mask_to_video(mask, save_path='mask.mp4', fps=24):
    T, _, H, W = mask.shape
    writer = imageio.get_writer(save_path, fps=fps, codec='libx264', quality=8)
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

class MaskInitType(Enum):
    system = auto()
    user = auto()

class BaseMaskGenerator(ABC):

    def __init__(self, mask_init_type=MaskInitType.system):
        if mask_init_type not in set(MaskInitType):
            raise ValueError('mask_init_type should be an instance of MaskInitType.')
        self.mask_init_type = mask_init_type

    def create_system_mask(self, num_frames, height, width, device, dtype):
        if num_frames is None or height is None or width is None:
            raise ValueError('num_frames, height, and width should be provided.')
        return torch.ones([num_frames, 1, height, width], device=device, dtype=dtype)

    def check_user_mask(self, mask):
        assert len(mask.shape) == 4, 'mask should have 4 dimensions.' 
        assert mask.shape[1] == 1, 'mask should have 1 channel.'
        # assert torch.all((tensor == 0) | (tensor == 1)), 'mask should be binary.' # annotate this to save time, but this condition requires user confirmation

    def process_for_null_mask(self, num_frames=None, height=None, width=None, device='cuda', dtype=torch.float32):
        raise ValueError('mask should not be None.')

    @abstractmethod
    def process(self, mask):
        # process self.mask to meet the specific task
        pass

    def __call__(self, num_frames=None, height=None, width=None, mask=None, device='cuda', dtype=torch.float32):
        if self.mask_init_type == MaskInitType.system:
            mask = self.create_system_mask(num_frames, height, width, device, dtype)
        elif self.mask_init_type == MaskInitType.user:
            if mask is not None:
                self.check_user_mask(mask)
            else:
                mask = self.process_for_null_mask(num_frames, height, width, device, dtype) 
        return self.process(mask)

class SemanticBaseMaskGenerator(BaseMaskGenerator):

    def __init__(self, mask_init_type=MaskInitType.user, min_clear_ratio_hw=0.0, max_clear_ratio_hw=1.0):
        super().__init__(mask_init_type=mask_init_type)
        self.generator_maybe_use = RandomSpatialMaskGenerator(min_clear_ratio_hw=min_clear_ratio_hw, max_clear_ratio_hw=max_clear_ratio_hw)
    
    @abstractmethod
    def process_for_semantic_inheritance(self, mask):
        pass
    
    def process_for_null_mask(self, num_frames, height, width, device, dtype):
        print('Mask is None, using random spatial mask as default.')
        return self.generator_maybe_use(num_frames, height, width, mask=None, device=device, dtype=dtype)

    def process(self, mask):
        if not torch.any(mask):
            num_frames, _, height, width = mask.shape
            device, dtype = mask.device, mask.dtype
            mask = self.generator_maybe_use(num_frames, height, width, mask=None, device=device, dtype=dtype)
        return self.process_for_semantic_inheritance(mask)

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
    
    def __init__(self, mask_init_type=MaskInitType.system, min_clear_ratio=0.0, max_clear_ratio=1.0):
        super().__init__(mask_init_type=mask_init_type)
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

    def __init__(self, mask_init_type=MaskInitType.system, min_clear_ratio=0.0, max_clear_ratio=1.0):
        super().__init__(mask_init_type=mask_init_type)
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


class SemanticMaskGenerator(SemanticBaseMaskGenerator):
    def process_for_semantic_inheritance(self, mask):
        return mask

    
class DilateMaskGenerator(SemanticMaskGenerator):

    def __init__(self, mask_init_type=MaskInitType.user, max_height=640, max_width=640, min_clear_ratio_hw=0.0, max_clear_ratio_hw=1.0, kernel_size=None):
        super().__init__(mask_init_type=mask_init_type, min_clear_ratio_hw=min_clear_ratio_hw, max_clear_ratio_hw=max_clear_ratio_hw)
        if max_height is not None and max_width is not None:
            self.kernel_size = min(max_height, max_width) // 40 * 2 + 1
        elif kernel_size is not None:
            assert kernel_size % 2 == 1 and kernel_size > 0, 'kernel_size should be an positive odd number.'
            self.kernel_size = kernel_size
        else:
            raise ValueError('kernel_size or max_height and max_width should be provided.')

    def process_for_semantic_inheritance(self, mask):
        num_frames = mask.shape[0]
        dtype, device = mask.dtype, mask.device
        mask = mask.cpu().numpy()
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        for t in range(num_frames):
            single_frame = mask[t, 0, :, :]
            dilated_frame = cv2.dilate(single_frame, kernel, iterations=1)
            mask[t, 0, :, :] = dilated_frame
        return torch.from_numpy(mask).to(dtype=dtype, device=device)

class BBoxMaskGenerator(SemanticMaskGenerator):

    def __init__(self, mask_init_type=MaskInitType.user, min_clear_ratio_hw=0.0, max_clear_ratio_hw=1.0):
        super().__init__(mask_init_type=mask_init_type, min_clear_ratio_hw=min_clear_ratio_hw, max_clear_ratio_hw=max_clear_ratio_hw)

    def generate_bbox(self, mask):
        num_frames, _, height, width = mask.shape  # C 应该为 1
        rows_with_foreground = torch.any(mask, dim=3).to(int).squeeze(1)  # (T, W)
        cols_with_foreground = torch.any(mask, dim=2).to(int).squeeze(1)  # (T, H)
        has_foreground = torch.any(rows_with_foreground, dim=1)
        min_row_indices = torch.where(has_foreground, torch.argmax(rows_with_foreground, dim=1), -1)  # (T,)
        max_row_indices = torch.where(
            has_foreground, 
            height - 1 - torch.argmax(torch.flip(rows_with_foreground, dims=[1]), dim=1), 
            -1
        )  # (T,)
        min_col_indices = torch.where(has_foreground, torch.argmax(cols_with_foreground, dim=1), -1)  # (T,)
        max_col_indices = torch.where(
            has_foreground, 
            width - 1 - torch.argmax(torch.flip(cols_with_foreground, dims=[1]), dim=1), 
            -1
        )  # (T,)

        return min_row_indices, min_col_indices, max_row_indices, max_col_indices

    def process_for_semantic_inheritance(self, mask):
        min_row_indices, min_col_indices, max_row_indices, max_col_indices = self.generate_bbox(mask)
        mask.zero_()
        for i, (min_row, min_col, max_row, max_col) in enumerate(zip(min_row_indices, min_col_indices, max_row_indices, max_col_indices)):
            if min_row != -1:
                mask[i, :, min_row:max_row+1, min_col:max_col+1].fill_(1)
        return mask

class RandomSpatialMaskGenerator(BaseMaskGenerator):

    def __init__(self, mask_init_type=MaskInitType.system, min_clear_ratio_hw=0.0, max_clear_ratio_hw=1.0):
        super().__init__(mask_init_type=mask_init_type)
        assert min_clear_ratio_hw >= 0 and min_clear_ratio_hw <= 1, 'min_clear_ratio_hw should be in the range of [0, 1].'
        assert max_clear_ratio_hw >= 0 and max_clear_ratio_hw <= 1, 'max_clear_ratio_hw should be in the range of [0, 1].'
        assert min_clear_ratio_hw <= max_clear_ratio_hw, 'min_clear_ratio_hw should be less than max_clear_ratio_hw.'
        self.min_clear_ratio_hw = min_clear_ratio_hw
        self.max_clear_ratio_hw = max_clear_ratio_hw

    def process(self, mask):
        height, width = mask.shape[2:]
        mask_height = random.randint(floor(height * self.min_clear_ratio_hw), ceil(height * self.max_clear_ratio_hw))
        mask_width = random.randint(floor(width * self.min_clear_ratio_hw), ceil(width * self.max_clear_ratio_hw))
        start_point_x = random.randint(0, height - mask_height)
        start_point_y = random.randint(0, width - mask_width)
        mask.zero_()
        mask[:, :, start_point_x:start_point_x + mask_height, start_point_y:start_point_y + mask_width].fill_(1)
        return mask

class OutpaintMaskGenerator(BaseMaskGenerator):

    def __init__(self):
        super().__init__(mask_init_type=MaskInitType.user)
    def process(self, mask):
        return 1 - mask

class MaskCompose:
    def __init__(self, mask_generators):
        self.mask_generators = mask_generators

    def __call__(self, num_frames=None, height=None, width=None, mask=None, device='cuda', dtype=torch.float32):
        mask = self.mask_generators[0](num_frames, height, width, mask, device, dtype)
        for generator in self.mask_generators[1:]:
            mask = generator(num_frames=None, height=None, width=None, mask=mask, device=device, dtype=dtype)
        return mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for m in self.mask_generators:
            format_string += "\n"
            format_string += f"    {m}"
        format_string += "\n)"
        return format_string
    
class MaskMixer:
    def __init__(self, mask_generator_temporal, mask_generator_spatial, operation=torch.logical_or):
        self.mask_generator_temporal = mask_generator_temporal
        self.mask_generator_spatial = mask_generator_spatial
        self.operation = operation
    
    def __call__(self, num_frames=None, height=None, width=None, mask=None, device='cuda', dtype=torch.float32):
        maskA = self.mask_generator_temporal(num_frames=num_frames, height=height, width=width, mask=None, device=device, dtype=dtype)
        maskB = self.mask_generator_spatial(num_frames=num_frames, height=height, width=width, mask=mask, device=device, dtype=dtype)
        return self.operation(maskA, maskB)
    
    def __repr__(self) -> str:
        return f'{self.mask_generator_spatial.__class__.__name__} {self.operation.__name__} {self.mask_generator_temporal.__class__.__name__}'
    

class MaskProcessor:
    def __init__(
        self, 
        max_height=640, 
        max_width=640, 
        min_clear_ratio=0.0, 
        max_clear_ratio=1.0, 
        min_clear_ratio_hw=0.1,
        max_clear_ratio_hw=0.9,
        dilate_kernel_size=31, 
    ):
        
        self.max_height = max_height
        self.max_width = max_width
        self.min_clear_ratio = min_clear_ratio
        self.max_clear_ratio = max_clear_ratio
        self.min_clear_ratio_hw = min_clear_ratio_hw
        self.max_clear_ratio_hw = max_clear_ratio_hw
        self.dilate_kernel_size = dilate_kernel_size

        self.init_mask_generators()

    def init_mask_generators(self):
        self.mask_generators = {
            MaskType.t2iv: T2IVMaskGenerator(),
            MaskType.i2v: I2VMaskGenerator(),
            MaskType.transition: TransitionMaskGenerator(),
            MaskType.continuation: ContinuationMaskGenerator(min_clear_ratio=self.min_clear_ratio, max_clear_ratio=self.max_clear_ratio),
            MaskType.clear: ClearMaskGenerator(),
            MaskType.random_temporal: RandomTemporalMaskGenerator(min_clear_ratio=self.min_clear_ratio, max_clear_ratio=self.max_clear_ratio),
            MaskType.semantic: SemanticMaskGenerator(min_clear_ratio_hw=self.min_clear_ratio_hw, max_clear_ratio_hw=self.max_clear_ratio_hw),
            MaskType.dilate: DilateMaskGenerator(
                max_height=self.max_height,
                max_width=self.max_width,
                min_clear_ratio_hw=self.min_clear_ratio_hw, 
                max_clear_ratio_hw=self.max_clear_ratio_hw,
                kernel_size=self.dilate_kernel_size,
            ),
            MaskType.bbox: BBoxMaskGenerator(min_clear_ratio_hw=self.min_clear_ratio_hw, max_clear_ratio_hw=self.max_clear_ratio_hw),
            MaskType.random_spatial: RandomSpatialMaskGenerator(min_clear_ratio_hw=self.min_clear_ratio_hw, max_clear_ratio_hw=self.max_clear_ratio_hw),
            MaskType.outpaint_random_spatial: MaskCompose([
                RandomSpatialMaskGenerator(min_clear_ratio_hw=self.min_clear_ratio_hw, max_clear_ratio_hw=self.max_clear_ratio_hw),
                OutpaintMaskGenerator()
            ]),
            MaskType.outpaint_semantic: MaskCompose([
                SemanticMaskGenerator(min_clear_ratio_hw=self.min_clear_ratio_hw, max_clear_ratio_hw=self.max_clear_ratio_hw), 
                OutpaintMaskGenerator()
            ]),
            MaskType.outpaint_bbox: MaskCompose([
                BBoxMaskGenerator(min_clear_ratio_hw=self.min_clear_ratio_hw, max_clear_ratio_hw=self.max_clear_ratio_hw), 
                OutpaintMaskGenerator()
            ]),
            MaskType.i2v_outpaint_semantic: MaskMixer(
                mask_generator_temporal=I2VMaskGenerator(), 
                mask_generator_spatial=MaskCompose([
                    SemanticMaskGenerator(min_clear_ratio_hw=self.min_clear_ratio_hw, max_clear_ratio_hw=self.max_clear_ratio_hw),
                    OutpaintMaskGenerator()
                ]),
                operation=torch.logical_or
            )
        }
    
    def __call__(self, pixel_values, mask=None, mask_type=None, mask_type_ratio_dict=None):

        num_frames, _, height, width = pixel_values.shape   

        if mask_type_ratio_dict is not None:  
            assert isinstance(mask_type_ratio_dict, dict), 'mask_type_ratio_dict should be a dict.'
            assert mask_type_ratio_dict.keys() <= set(MaskType), f'Invalid mask type: {set(MaskType) - mask_type_ratio_dict.keys()}'
            mask_generator_type = random.choices(list(mask_type_ratio_dict.keys()), list(mask_type_ratio_dict.values()))[0]
        elif mask_type is not None:
            assert mask_type in MaskType or mask_type in STR_TO_TYPE.keys(), f'Invalid mask type: {mask_type}'
            mask_generator_type = mask_type if mask_type in MaskType else STR_TO_TYPE[mask_type]
        else:
            raise ValueError('mask_type or mask_type_ratio_dict should be provided.')
        
        mask = self.mask_generators[mask_generator_type](num_frames, height, width, mask, device=pixel_values.device, dtype=pixel_values.dtype)

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
        mask = mask.transpose(1, 2).contiguous()
        return mask
    

if __name__ == '__main__':
    video_path = '/home/image_data/hxy/data/video/000184.mp4'
    semantic_mask_path = '/home/image_data/gyy/suv/Open-Sora-Plan/mask0000.mp4'
    bbox_mask_path = '/home/image_data/hxy/data/video/000001_bbox.mp4'
    video = read_video(video_path)
    mask = read_video(semantic_mask_path)
    mask = mask[:, 0]
    mask = (mask > 128).unsqueeze(1).to(torch.float32)
    processor = MaskProcessor()
    ratio_dict = {
        MaskType.t2iv: 0,
        MaskType.i2v: 0,
        MaskType.transition: 0,
        MaskType.continuation: 0,
        MaskType.clear: 0,
        MaskType.random_temporal: 0,
        MaskType.semantic: 1,
        MaskType.dilate: 0,
        MaskType.bbox: 0,
        MaskType.random_spatial: 0,
        MaskType.outpaint_random_spatial: 0,
        MaskType.outpaint_semantic: 0,
        MaskType.outpaint_bbox: 0,
        MaskType.i2v_outpaint_semantic: 0
    }

    import yaml
    with open('/home/image_data/gyy/suv/Open-Sora-Plan/scripts/train_configs/mask_config.yaml', 'r') as f:
        yaml_config = yaml.safe_load(f)
    print(yaml_config)

    mask = processor(mask, mask=mask, mask_type_ratio_dict=ratio_dict)['mask']
    print(mask.shape)
    save_mask_to_video(mask, save_path='dilate_mask.mp4', fps=24)
    
    