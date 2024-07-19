from torchvision.transforms import Compose
from transformers import AutoTokenizer, AutoImageProcessor

from torchvision import transforms
from torchvision.transforms import Lambda

# try:
#     import torch_npu
#     from opensora.npu_config import npu_config
# from .t2v_datasets_npu import T2V_dataset
# except:
#     torch_npu = None
#     npu_config = None
#     from .t2v_datasets import T2V_dataset
from opensora.dataset.t2v_datasets import T2V_dataset
from opensora.dataset.inpaint_datasets import Inpaint_dataset
from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo


ae_norm = {
    'CausalVAEModel_D8_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVQVAEModel_4x4x4': Lambda(lambda x: x - 0.5),
    'CausalVQVAEModel_4x8x8': Lambda(lambda x: x - 0.5),
    'VQVAEModel_4x4x4': Lambda(lambda x: x - 0.5),
    'VQVAEModel_4x8x8': Lambda(lambda x: x - 0.5),
    "bair_stride4x2x2": Lambda(lambda x: x - 0.5),
    "ucf101_stride4x4x4": Lambda(lambda x: x - 0.5),
    "kinetics_stride4x4x4": Lambda(lambda x: x - 0.5),
    "kinetics_stride2x4x4": Lambda(lambda x: x - 0.5),
    'stabilityai/sd-vae-ft-mse': transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    'stabilityai/sd-vae-ft-ema': transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    'vqgan_imagenet_f16_1024': Lambda(lambda x: 2. * x - 1.),
    'vqgan_imagenet_f16_16384': Lambda(lambda x: 2. * x - 1.),
    'vqgan_gumbel_f8': Lambda(lambda x: 2. * x - 1.),

}
ae_denorm = {
    'CausalVAEModel_D8_4x8x8': lambda x: (x + 1.) / 2.,
    'CausalVAEModel_2x8x8': lambda x: (x + 1.) / 2.,
    'CausalVAEModel_4x8x8': lambda x: (x + 1.) / 2.,
    'CausalVQVAEModel_4x4x4': lambda x: x + 0.5,
    'CausalVQVAEModel_4x8x8': lambda x: x + 0.5,
    'VQVAEModel_4x4x4': lambda x: x + 0.5,
    'VQVAEModel_4x8x8': lambda x: x + 0.5,
    "bair_stride4x2x2": lambda x: x + 0.5,
    "ucf101_stride4x4x4": lambda x: x + 0.5,
    "kinetics_stride4x4x4": lambda x: x + 0.5,
    "kinetics_stride2x4x4": lambda x: x + 0.5,
    'stabilityai/sd-vae-ft-mse': lambda x: 0.5 * x + 0.5,
    'stabilityai/sd-vae-ft-ema': lambda x: 0.5 * x + 0.5,
    'vqgan_imagenet_f16_1024': lambda x: (x + 1.) / 2.,
    'vqgan_imagenet_f16_16384': lambda x: (x + 1.) / 2.,
    'vqgan_gumbel_f8': lambda x: (x + 1.) / 2.,
}

def getdataset(args):
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = ae_norm[args.ae]
    if args.dataset == 't2v':
        transform = transforms.Compose([
            ToTensorVideo(),
            LongSideResizeVideo((args.max_height, args.max_width), skip_low_resolution=True), 
            SpatialStrideCropVideo(stride=args.hw_stride), 
            norm_fun
        ])
        tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
        return T2V_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer)
    elif args.dataset == 'i2v' or args.dataset == 'inpaint':
        resize_topcrop = [CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True), ]
        # if args.multi_scale:
        #     resize = [
        #         LongSideResizeVideo(args.max_image_size, skip_low_resolution=True),
        #         SpatialStrideCropVideo(args.stride)
        #         ]
        # else:
        resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
        transform = transforms.Compose([
            ToTensorVideo(),
            *resize, 
            # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
            norm_fun
        ])
        transform_topcrop = transforms.Compose([
            ToTensorVideo(),
            *resize_topcrop, 
            # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
            norm_fun
        ])
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
        return Inpaint_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer, 
                           transform_topcrop=transform_topcrop)
    raise NotImplementedError(args.dataset)


if __name__ == "__main__":
    from accelerate import Accelerator
    from opensora.dataset.t2v_datasets import dataset_prog
    import random
    from tqdm import tqdm
    args = type('args', (), 
    {
        'ae': 'CausalVAEModel_4x8x8', 
        'dataset': 't2v', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 320,
        'max_width': 240,
        'hw_stride': 32, 
        'skip_low_resolution': True, 
        'num_frames': 330,
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        'cache_dir': '../cache_dir', 
        'data': 'scripts/train_data/merge_data.txt', 
        'train_fps': 24, 
        'drop_short_ratio': 0.0, 
        'use_img_from_vid': False, 
        'speed_factor': 1.0, 
        'cfg': 0.1, 
        'text_encoder_name': 'google/mt5-xxl', 
        'dataloader_num_workers': 10,

    }
    )
    accelerator = Accelerator()
    dataset = getdataset(args)
    data = next(iter(dataset))
    import ipdb;ipdb.set_trace()
    print()