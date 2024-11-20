from torchvision.transforms import Compose
from transformers import AutoTokenizer, AutoImageProcessor

from torchvision import transforms
from torchvision.transforms import Lambda

try:
    import torch_npu
except:
    torch_npu = None

from opensora.dataset.t2v_datasets import T2V_dataset
from opensora.dataset.inpaint_dataset import Inpaint_dataset
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, MaxHWStrideResizeVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize


from accelerate.logging import get_logger
logger = get_logger(__name__)

def getdataset(args):
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = ae_norm[args.ae]
    if args.force_resolution:
        resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
    else:
        resize = [
            MaxHWStrideResizeVideo(args.max_hxw, force_5_ratio=args.force_5_ratio, hw_stride=args.hw_stride, interpolation_mode="bicubic"), 
            SpatialStrideCropVideo(stride=args.hw_stride, force_5_ratio=args.force_5_ratio), 
        ]

    # tokenizer_1 = AutoTokenizer.from_pretrained(args.text_encoder_name_1, cache_dir=args.cache_dir)
    tokenizer_1 = AutoTokenizer.from_pretrained('/storage/cache_dir/t5-v1_1-xl', cache_dir=args.cache_dir)
    # tokenizer_1 = AutoTokenizer.from_pretrained('/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl', cache_dir=args.cache_dir)
    tokenizer_2 = None
    if args.text_encoder_name_2 is not None:
        # tokenizer_2 = AutoTokenizer.from_pretrained(args.text_encoder_name_2, cache_dir=args.cache_dir)
        tokenizer_2 = AutoTokenizer.from_pretrained('/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k', cache_dir=args.cache_dir)
    if args.dataset == 't2v':
        transform = transforms.Compose([
            ToTensorVideo(),
            *resize, 
            norm_fun
        ])  # also work for img, because img is video when frame=1
        return T2V_dataset(
            args, transform=transform, temporal_sample=temporal_sample, 
            tokenizer_1=tokenizer_1, tokenizer_2=tokenizer_2
            )
    elif args.dataset == 'i2v' or args.dataset == 'inpaint':
        resize_transform = Compose(resize)
        transform = Compose([
            ToTensorAfterResize(),
            norm_fun,
        ])
        return Inpaint_dataset(
            args, resize_transform=resize_transform, transform=transform, 
            temporal_sample=temporal_sample, tokenizer_1=tokenizer_1, tokenizer_2=tokenizer_2
        )
    raise NotImplementedError(args.dataset)


if __name__ == "__main__":
    '''
    python opensora/dataset/__init__.py
    accelerate launch --num_processes 1 opensora/dataset/__init__.py
    '''
    from accelerate import Accelerator
    from opensora.utils.dataset_utils import LengthGroupedSampler, Collate
    from torch.utils.data import DataLoader
    import random
    from torch import distributed as dist
    from tqdm import tqdm
    import imageio
    import numpy as np
    from einops import rearrange

    args = type('args', (), 
    {
        'ae': 'WFVAEModel_D32_8x8x8', 
        'dataset': 't2v', 
        'model_max_length': 512, 
        'max_height': 768,
        'max_width': 768,
        'hw_stride': 16, 
        'num_frames': 105,
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        'cache_dir': '../cache_dir', 
        'data': '/storage/ongoing/9.29/mmdit/1.5/Open-Sora-Plan/scripts/train_data/merge_data.txt', 
        'train_fps': 18, 
        'drop_short_ratio': 0.0, 
        'speed_factor': 1.0, 
        'cfg': 0.1, 
        'text_encoder_name_1': 'google/mt5-xxl', 
        'text_encoder_name_2': None,
        'dataloader_num_workers': 8,
        'force_resolution': False, 
        'use_decord': True, 
        'group_data': True, 
        'train_batch_size': 1, 
        'gradient_accumulation_steps': 1, 
        'ae_stride': 8, 
        'ae_stride_t': 8,  
        'patch_size': 2, 
        'patch_size_t': 1, 
        'total_batch_size': 256, 
        'sp_size': 1, 
        'max_hxw': 384*384, 
        'min_hxw': 384*288, 
        'force_5_ratio': True, 
        'random_data': False, 
        'train_image_batch_size': 1
    }
    )
    accelerator = Accelerator()
    dataset = getdataset(args)
    import ipdb;ipdb.set_trace()
    sampler = LengthGroupedSampler(
                args.train_batch_size,
                world_size=1, 
                gradient_accumulation_size=args.gradient_accumulation_steps, 
                initial_global_step=0, 
                lengths=dataset.lengths, 
                group_data=args.group_data, 
            )
    train_dataloader = DataLoader(
        dataset,
        shuffle=False,
        # pin_memory=True,
        collate_fn=Collate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        sampler=sampler, 
        drop_last=False, 
        prefetch_factor=4
    )
    for idx, i in enumerate(tqdm(train_dataloader)):
        import ipdb;ipdb.set_trace()
        pixel_values = i[0][0]
        pixel_values_ = (pixel_values+1)/2
        pixel_values_ = rearrange(pixel_values_, 'c t h w -> t h w c') * 255.0
        pixel_values_ = pixel_values_.numpy().astype(np.uint8)
        imageio.mimwrite(f'output{idx}.mp4', pixel_values_, fps=args.train_fps)