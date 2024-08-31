from torchvision.transforms import Compose
from transformers import AutoTokenizer, AutoImageProcessor

from torchvision import transforms
from torchvision.transforms import Lambda



from opensora.dataset.t2v_datasets import T2V_dataset
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize

from opensora.dataset.inpaint_datasets import Inpaint_dataset


def getdataset(args):
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = ae_norm[args.ae]
    if args.force_resolution:
        resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
    else:
        resize = [
            LongSideResizeVideo((args.max_height, args.max_width), skip_low_resolution=True), 
            SpatialStrideCropVideo(stride=args.hw_stride), 
        ]
    transform = transforms.Compose([
        ToTensorVideo(),
        *resize, 
        norm_fun
    ])
    tokenizer = AutoTokenizer.from_pretrained("/home/image_data/mt5-xxl", cache_dir=args.cache_dir)
    # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/models--DeepFloyd--t5-v1_1-xxl/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37", cache_dir=args.cache_dir)
    # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    if args.dataset == 't2v':
        return T2V_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer)
    elif args.dataset == 'inpaint' or args.dataset == 'i2v':
        mask_processor = transforms.Compose([*resize])
        return Inpaint_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer, mask_processor=mask_processor)
    raise NotImplementedError(args.dataset)


if __name__ == "__main__":
    from accelerate import Accelerator
    from opensora.dataset.t2v_datasets import dataset_prog
    from opensora.utils.dataset_utils import LengthGroupedSampler, Collate
    from torch.utils.data import DataLoader
    import random
    from torch import distributed as dist
    from tqdm import tqdm
    args = type('args', (), 
    {
        'ae': 'CausalVAEModel_D8_4x8x8', 
        'dataset': 't2v', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 320,
        'max_width': 320,
        'hw_stride': 32, 
        'skip_low_resolution': True, 
        'num_frames': 93,
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        'cache_dir': '../cache_dir', 
        'data': 'scripts/train_data/merge_data_debug.txt', 
        'train_fps': 16, 
        'drop_short_ratio': 0.0, 
        'use_img_from_vid': False, 
        'speed_factor': 1.0, 
        'cfg': 0.1, 
        'text_encoder_name': 'google/mt5-xxl', 
        'dataloader_num_workers': 10,
        'use_motion': False, 
        'force_resolution': False, 
        'use_decord': True, 
        'group_data': True, 
        'train_batch_size': 1, 
        'gradient_accumulation_steps': 1, 
        'ae_stride': 8, 
        'ae_stride_t': 4, 
        'patch_size': 2, 
        'patch_size_t': 1, 
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