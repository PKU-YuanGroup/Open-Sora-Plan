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
from .t2v_datasets import T2V_dataset
from .inpaint_datasets import Inpaint_dataset
from .videoip_datasets_bak import VideoIP_dataset
from .transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo


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
        transform_topcrop = transforms.Compose([
            ToTensorVideo(),
            *resize_topcrop, 
            # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
            norm_fun
        ])
        # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/models--DeepFloyd--t5-v1_1-xxl/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37", cache_dir=args.cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/models--google--mt5-xl/snapshots/63fc6450d80515b48e026b69ef2fbbd426433e84", cache_dir=args.cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
        return T2V_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer, 
                           transform_topcrop=transform_topcrop)
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
        transform_topcrop = transforms.Compose([
            ToTensorVideo(),
            *resize_topcrop, 
            # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
            norm_fun
        ])
        # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
        return Inpaint_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer, 
                           transform_topcrop=transform_topcrop)
    elif args.dataset == 'vip':
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
        transform_topcrop = transforms.Compose([
            ToTensorVideo(),
            *resize_topcrop, 
            # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
            norm_fun
        ])
        # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
        return Inpaint_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer, 
                           transform_topcrop=transform_topcrop)
    raise NotImplementedError(args.dataset)
