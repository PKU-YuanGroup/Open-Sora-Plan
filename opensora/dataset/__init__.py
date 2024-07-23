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
from .inpaint_datasets import get_inpaint_dataset
from .transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize
from opensora.models.diffusion.opensora.modeling_inpaint import ModelType, STR_TO_TYPE, TYPE_TO_STR

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
        # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/models--DeepFloyd--t5-v1_1-xxl/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37", cache_dir=args.cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/models--google--mt5-xl/snapshots/63fc6450d80515b48e026b69ef2fbbd426433e84", cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
        return T2V_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer, 
                           transform_topcrop=transform_topcrop)
    elif args.dataset == 'inpaint' or args.dataset == 'i2v' or args.dataset == 'vip':
        resize_topcrop = CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True)
        resize = CenterCropResizeVideo((args.max_height, args.max_width))
        transform = transforms.Compose([
            ToTensorAfterResize(),
            # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
            norm_fun
        ])
        transform_topcrop = transforms.Compose([
            ToTensorAfterResize(),
            # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
            norm_fun
        ])

        if STR_TO_TYPE[args.model_type] != ModelType.INPAINT_ONLY:
            if args.max_width / args.max_height == 4 / 3:
                image_processor_center_crop = transforms.CenterCrop((518, 686))
            elif args.max_width / args.max_height == 16 / 9:
                image_processor_center_crop = transforms.CenterCrop((518, 910))
            else:
                image_processor_center_crop = transforms.CenterCrop((518, 518))

            # dino image processor
            image_processor = transforms.Compose([
                transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True, max_size=None),
                image_processor_center_crop, #
                ToTensorAfterResize(),
                transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
            ])
        else:
            image_processor = None
        # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)

        dataset = get_inpaint_dataset(args.model_type)

        return dataset(args, transform=transform, resize_transform=resize, resize_transform_topcrop=resize_topcrop, temporal_sample=temporal_sample, tokenizer=tokenizer, 
                           transform_topcrop=transform_topcrop, image_processor=image_processor)
    
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
        'num_frames': 1,
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        'cache_dir': '../cache_dir', 
        'image_data': '/storage/ongoing/new/Open-Sora-Plan-bak/7.14bak/scripts/train_data/image_data.txt', 
        'video_data': '1',
        'train_fps': 24, 
        'drop_short_ratio': 1.0, 
        'use_img_from_vid': False, 
        'speed_factor': 1.0, 
        'cfg': 0.1, 
        'text_encoder_name': 'google/mt5-xxl', 
        'dataloader_num_workers': 10,

    }
    )
    accelerator = Accelerator()
    dataset = getdataset(args)
    num = len(dataset_prog.img_cap_list)
    zero = 0
    for idx in tqdm(range(num)):
        image_data = dataset_prog.img_cap_list[idx]
        caps = [i['cap'] if isinstance(i['cap'], list) else [i['cap']] for i in image_data]
        try:
            caps = [[random.choice(i)] for i in caps]
        except Exception as e:
            print(e)
            # import ipdb;ipdb.set_trace()
            print(image_data)
            zero += 1
            continue
        assert caps[0] is not None and len(caps[0]) > 0
    print(num, zero)
    import ipdb;ipdb.set_trace()
    print('end')