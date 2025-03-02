import copy
import torchvision.transforms as transforms

from mindspeed_mm.data.data_utils.data_transform import (
    AENorm,
    CenterCropResizeVideo,
    LongSideResizeVideo,
    RandomHorizontalFlipVideo,
    SpatialStrideCropVideo,
    ToTensorVideo,
    ToTensorAfterResize,
    Expand2Square,
    JpegDegradationSimulator,
    MaxHWResizeVideo,
    MaxHWStrideResizeVideo,
    SpatialStrideCropVideo,
)

VIDEO_TRANSFORM_MAPPING = {
    "ToTensorVideo": ToTensorVideo,
    "ToTensorAfterResize": ToTensorAfterResize,
    "RandomHorizontalFlipVideo": RandomHorizontalFlipVideo,
    "CenterCropResizeVideo": CenterCropResizeVideo,
    "LongSideResizeVideo": LongSideResizeVideo,
    "MaxHWStrideResizeVideo": MaxHWStrideResizeVideo,
    "MaxHWResizeVideo": MaxHWResizeVideo,
    "SpatialStrideCropVideo": SpatialStrideCropVideo,
    "norm_fun": transforms.Normalize,
    "ae_norm": AENorm,
}


IMAGE_TRANSFORM_MAPPING = {
    "Lambda": transforms.Lambda,
    "ToTensorVideo": ToTensorVideo,
    "ToTensorAfterResize": ToTensorAfterResize,
    "CenterCropResizeVideo": CenterCropResizeVideo,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "RandomHorizontalFlipVideo": RandomHorizontalFlipVideo,
    "MaxHWStrideResizeVideo": MaxHWStrideResizeVideo,
    "MaxHWResizeVideo": MaxHWResizeVideo,
    "SpatialStrideCropVideo": SpatialStrideCropVideo,
    "ToTensor": transforms.ToTensor,
    "norm_fun": transforms.Normalize,
    "ae_norm": AENorm,
    "DataAugment": JpegDegradationSimulator,
    "Pad2Square": Expand2Square,
    "Resize": transforms.Resize,
}


INTERPOLATIONMODE_LIST = [
    "bicubic",
    "bilinear",
    "nearest",
    "nearest-exact",
    "box",
    "hamming",
    "lanczos",
]



def get_transforms(is_video=True, train_pipeline=None):
    if train_pipeline is None:
        return None
    train_pipeline_info = (
        copy.deepcopy(train_pipeline.get("video", list()))
        if is_video
        else copy.deepcopy(train_pipeline.get("image", list()))
    )
    pipeline = []
    for pp_in in train_pipeline_info:
        param_info = pp_in.get("param", dict())
        trans_type = pp_in.get("trans_type", "")
        trans_info = TransformMaping(
            is_video=is_video, trans_type=trans_type, param=param_info
        ).get_trans_func()
        pipeline.append(trans_info)
    output_transforms = transforms.Compose(pipeline)
    return output_transforms

class TransformMaping:
    """used for transforms mapping"""

    def __init__(self, is_video=True, trans_type="", param=None):
        self.is_video = is_video
        self.trans_type = trans_type
        self.param = param if param is not None else dict()

    def get_trans_func(self):
        if self.is_video:
            if self.trans_type in VIDEO_TRANSFORM_MAPPING:
                transforms_cls = VIDEO_TRANSFORM_MAPPING[self.trans_type]
                if "Resize" in self.trans_type  and "interpolation_mode" in self.param:
                    if self.param["interpolation_mode"] not in INTERPOLATIONMODE_LIST:
                        raise ValueError(
                            f"Unsupported interpolation mode: {self.param['interpolation_mode']}"
                        )
                return transforms_cls(**self.param)
            else:
                raise NotImplementedError(
                    f"Unsupported video transform type: {self.trans_type}"
                )
        else:
            if self.trans_type in IMAGE_TRANSFORM_MAPPING:
                transforms_cls = IMAGE_TRANSFORM_MAPPING[self.trans_type]
                if "Resize" in self.trans_type  and "interpolation_mode" in self.param:
                    if self.param["interpolation_mode"] not in INTERPOLATIONMODE_LIST:
                        raise ValueError(
                            f"Unsupported interpolation mode: {self.param['interpolation_mode']}"
                        )
                return transforms_cls(**self.param)
            else:
                raise NotImplementedError(
                    f"Unsupported image transform type: {self.trans_type}"
                )
