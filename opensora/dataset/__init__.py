from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import ToTensorVideo, RandomHorizontalFlipVideo

from opensora.models.ae import vae, vqvae, videovae, videovqvae
from .feature_datasets import LandscopeFeatures
from torchvision import transforms
from torchvision.transforms import Lambda

from .transform import LongSideScale, TemporalRandomCrop
from .ucf101 import UCF101


def getdataset(args):
    temporal_sample = TemporalRandomCrop(args.num_frames * args.sample_rate)  # 16 x
    if args.ae in videovqvae:
        norm_fun = Lambda(lambda x: x - 0.5)
    elif args.ae in videovae:
        raise NotImplementedError
    elif args.ae in vae:
        norm_fun = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    elif args.ae in vqvae:
        norm_fun = Lambda(lambda x: 2. * x - 1.)
    else:
        raise NotImplementedError

    if args.dataset == 'landscope_feature':
        temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 1
        return LandscopeFeatures(args, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101':
        transform = Compose(
            [
                ToTensorVideo(),  # cthw
                LongSideScale(size=args.max_image_size),
                RandomHorizontalFlipVideo(p=0.5),
                norm_fun,
            ]
        )
        return UCF101(args, transform=transform, temporal_sample=temporal_sample)