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
from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, MaxHWResizeVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize



def getdataset(args):
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = ae_norm[args.ae]
    if args.force_resolution:
        resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
    else:
        resize = [
            MaxHWResizeVideo(args.max_hxw), 
            SpatialStrideCropVideo(stride=args.hw_stride), 
        ]

    tokenizer_1 = AutoTokenizer.from_pretrained(args.text_encoder_name_1, cache_dir=args.cache_dir)
    tokenizer_2 = None
    if args.text_encoder_name_2 is not None:
        tokenizer_2 = AutoTokenizer.from_pretrained(args.text_encoder_name_2, cache_dir=args.cache_dir)
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
    '''
    from accelerate import Accelerator
    from opensora.dataset.t2v_datasets import dataset_prog
    from opensora.utils.dataset_utils import LengthGroupedSampler, Collate
    from torch.utils.data import DataLoader
    import random
    from torch import distributed as dist
    from tqdm import tqdm
    args = type('args', (), 
    {
        'ae': 'WFVAEModel_D32_4x8x8', 
        'dataset': 't2v', 
        'model_max_length': 512, 
        'max_height': 640,
        'max_width': 640,
        'hw_stride': 16, 
        'num_frames': 93,
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        'cache_dir': '../cache_dir', 
        'data': '/home/image_data/gyy/mmdit/Open-Sora-Plan/scripts/train_data/current_hq_on_npu.txt', 
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
        'ae_stride_t': 4,  
        'patch_size': 2, 
        'patch_size_t': 1, 
        'total_batch_size': 256, 
        'sp_size': 1, 
        'max_hxw': 384*384, 
        'min_hxw': 384*288, 
        # 'max_hxw': 236544, 
        # 'min_hxw': 102400, 
    }
    )
    # accelerator = Accelerator()
    dataset = getdataset(args)
    # data = next(iter(dataset))
    # import ipdb;ipdb.set_trace()
    # print()
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
    import ipdb;ipdb.set_trace()
    import imageio
    import numpy as np
    from einops import rearrange
    while True:
        for idx, i in enumerate(tqdm(train_dataloader)):
            pixel_values = i[0][0]
            pixel_values_ = (pixel_values+1)/2
            pixel_values_ = rearrange(pixel_values_, 'c t h w -> t h w c') * 255.0
            pixel_values_ = pixel_values_.numpy().astype(np.uint8)
            imageio.mimwrite(f'output{idx}.mp4', pixel_values_, fps=args.train_fps)
            dist.barrier()
            pass