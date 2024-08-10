from torchvision.transforms import Compose
from transformers import AutoTokenizer, AutoImageProcessor

from torchvision import transforms
from torchvision.transforms import Lambda

from opensora.dataset.t2v_datasets import T2V_dataset
from opensora.dataset.inpaint_datasets import Inpaint_dataset
from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo
from opensora.models.causalvideovae import ae_norm, ae_denorm

def getdataset(args):
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = ae_norm[args.ae]
    resize_topcrop = [CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True), ]
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
    # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    if args.dataset == 't2v':
        return T2V_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer, 
                           transform_topcrop=transform_topcrop)
    elif args.dataset == 'inpaint' or args.dataset == 'i2v':
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