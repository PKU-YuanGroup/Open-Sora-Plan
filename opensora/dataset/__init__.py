from torchvision.transforms import Compose
from transformers import AutoTokenizer, AutoImageProcessor

from torchvision import transforms
from torchvision.transforms import Lambda



from opensora.dataset.t2v_datasets import T2V_dataset
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize



def getdataset(args):
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = ae_norm[args.ae]
    if args.force_resolution:
        resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
        resize_for_img = None
        assert args.max_height_for_img is None
        assert args.max_width_for_img is None
    else:
        assert (args.min_height is not None and args.skip_low_resolution) or (args.min_height is None)
        assert (args.min_width is not None and args.skip_low_resolution) or (args.min_width is None)
        resize = [
            LongSideResizeVideo((args.max_height, args.max_width), skip_low_resolution=args.skip_low_resolution), 
            SpatialStrideCropVideo(stride=args.hw_stride), 
        ]
        resize_for_img = None
        if args.max_height_for_img is not None and args.max_width_for_img is not None:
            assert args.max_height_for_img > 0 and args.max_width_for_img > 0
            assert args.max_height_for_img > args.max_height and args.max_width_for_img > args.max_width
            resize_for_img = [
                LongSideResizeVideo((args.max_height_for_img, args.max_width_for_img), skip_low_resolution=args.skip_low_resolution), 
                SpatialStrideCropVideo(stride=args.hw_stride), 
            ]

    transform = transforms.Compose([
        ToTensorVideo(),
        *resize, 
        norm_fun
    ])  # also work for img, because img is video when frame=1

    transform_img = None
    if resize_for_img is not None:
        transform_img = transforms.Compose([
            ToTensorVideo(),
            *resize_for_img, 
            norm_fun
        ])

    # tokenizer_1 = AutoTokenizer.from_pretrained(args.text_encoder_name_1, cache_dir=args.cache_dir)
    # tokenizer_1 = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
    tokenizer_1 = AutoTokenizer.from_pretrained('/storage/cache_dir/t5-v1_1-xl', cache_dir=args.cache_dir)
    tokenizer_2 = None
    if args.text_encoder_name_2 is not None:
        # tokenizer_2 = AutoTokenizer.from_pretrained(args.text_encoder_name_2, cache_dir=args.cache_dir)
        tokenizer_2 = AutoTokenizer.from_pretrained('/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k', cache_dir=args.cache_dir)
    if args.dataset == 't2v':
        return T2V_dataset(
            args, transform=transform, transform_img=transform_img, 
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
        'ae': 'WFVAEModel_D32_8x8x8', 
        'dataset': 't2v', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 384,
        'max_width': 384,
        'hw_stride': 16, 
        'skip_low_resolution': True, 
        'num_frames': 93,
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        'cache_dir': '../cache_dir', 
        'data': 'scripts/train_data/image_data_debug.txt', 
        'train_fps': 16, 
        'drop_short_ratio': 0.0, 
        'use_img_from_vid': False, 
        'speed_factor': 1.0, 
        'cfg': 0.1, 
        'text_encoder_name_1': 'google/mt5-xxl', 
        'text_encoder_name_2': 'google/mt5-xxl', 
        'dataloader_num_workers': 10,
        'use_motion': False, 
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
        'min_height': 160, 
        'min_width': 160, 
        'ood_img_ratio': 0, 
        'sp_size': 1, 
        'max_height_for_img': None, 
        'max_width_for_img': None, 
    }
    )
    accelerator = Accelerator()
    dataset = getdataset(args)
    # data = next(iter(dataset))
    # import ipdb;ipdb.set_trace()
    # print()
    sampler = LengthGroupedSampler(
                args.train_batch_size,
                world_size=accelerator.num_processes, 
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
    import imageio
    import numpy as np
    from einops import rearrange
    while True:
        for idx, i in enumerate(tqdm(train_dataloader)):
            import ipdb;ipdb.set_trace()
            pixel_values = i[0][0]
            pixel_values_ = (pixel_values+1)/2
            pixel_values_ = rearrange(pixel_values_, 'c t h w -> t h w c') * 255.0
            pixel_values_ = pixel_values_.numpy().astype(np.uint8)
            imageio.mimwrite(f'output{idx}.mp4', pixel_values_, fps=args.train_fps)
            dist.barrier()
            pass