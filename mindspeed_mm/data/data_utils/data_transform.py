# Modified from Latte: https://github.com/Vchitect/Latte/blob/main/datasets/video_transforms.py

import io
import random
import numbers
import math

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("Clip should be Tensor, but it is %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("Clip should be 4D, but it is %dD" % clip.dim())

    return True


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    )


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError(
            "Clip tensor should have data type uint8, but it is %s" % str(clip.dtype)
        )
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]


def resize(clip, target_size, interpolation_mode, align_corners=False, antialias=False):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip,
        size=target_size,
        mode=interpolation_mode,
        align_corners=align_corners,
        antialias=antialias,
    )


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    H, W = clip.size(-2), clip.size(-1)
    scale = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(
        clip, scale_factor=scale, mode=interpolation_mode, align_corners=False
    )


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)


def center_crop_th_tw(clip, th, tw, top_crop):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")

    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)

    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)


def resize_crop_to_fill(clip, target_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = target_size[0], target_size[1]
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        clip = resize(clip, (sh, sw), "bilinear")
        i = 0
        j = int(round(sw - tw) / 2.0)
    else:
        sh, sw = round(h * rw), tw
        clip = resize(clip, (sh, sw), "bilinear")
        i = int(round(sh - th) / 2.0)
        j = 0
    if i + th > clip.size(-2) or j + tw > clip.size(-1):
        raise AssertionError("size mismatch.")
    return crop(clip, i, j, th, tw)


class AENorm:
    """
    Apply an ae_norm to a PIL image or video.
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Apply the center crop to the input video.

        Args:
            video (clip): The input video.

        Returns:
            video: The ae_norm video.
        """

        clip = 2.0 * clip - 1.0
        return clip

    def __repr__(self) -> str:
        return self.__class__.__name__


class CenterCropArr:
    """
    Apply a center crop to a PIL image.
    """

    def __init__(self, size=256):
        self.size = size

    def __call__(self, pil_image):
        """
        Apply the center crop to the input PIL image.

        Args:
            pil_image (PIL.Image): The input PIL image.

        Returns:
            PIL.Image: The center-cropped image.
        """
        return center_crop_arr(pil_image, self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ResizeCropToFill:
    """
    Apply a resize crop to a PIL image.
    """

    def __init__(self, size=256):
        self.size = size

    def __call__(self, pil_image):
        """
        Apply the resize crop to the input PIL image.

        Args:
            pil_image (PIL.Image): The input PIL image.

        Returns:
            PIL.Image: The resize-cropped image.
        """
        return resize_crop_to_fill(pil_image, self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ResizeCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        clip = resize_crop_to_fill(clip, self.size)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (T, C, H, W)
        Return:
            clip (torch.tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class SpatialStrideCropVideo:
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: cropped video clip by stride.
                size is (T, C, OH, OW)
        """
        # p_l: i, j, h, w
        p_l = self.get_params(clip)
        return crop(clip, p_l[0], p_l[1], p_l[2], p_l[3])

    def get_params(self, clip):
        h, w = clip.shape[-2:]
        th, tw = h // self.stride * self.stride, w // self.stride * self.stride
        # from top-left
        param_list = [0, 0, th, tw]
        return param_list

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stride={self.stride})"


class LongSideResizeVideo:
    """
    First use the long side,
    then resize to the specified size
    """

    def __init__(
            self,
            size,
            skip_low_resolution=False,
            interpolation_mode="bilinear",
    ):
        self.size = size
        self.skip_low_resolution = skip_low_resolution
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
                size is (T, C, 512, *) or (T, C, *, 512)
        """
        _, _, h, w = clip.shape
        if self.skip_low_resolution and max(h, w) <= self.size:
            return clip
        if h > w:
            w = int(w * self.size / h)
            h = self.size
        else:
            h = int(h * self.size / w)
            w = self.size
        resize_clip = resize(
            clip, target_size=(h, w), interpolation_mode=self.interpolation_mode
        )
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class CenterCropResizeVideo:
    """
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    """

    def __init__(
            self,
            size,
            use_short_edge=False,
            top_crop=False,
            interpolation_mode="bilinear",
            align_corners=False,
            antialias=False,
    ):
        if isinstance(size, list):
            size = tuple(size)
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)

        self.use_short_edge = use_short_edge
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        if self.use_short_edge:
            clip_center_crop = center_crop_using_short_edge(clip)
        else:
            clip_center_crop = center_crop_th_tw(
                clip, self.size[0], self.size[1], top_crop=self.top_crop
            )

        clip_center_crop_resize = resize(
            clip_center_crop,
            target_size=self.size,
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class UCFCenterCropVideo:
    """
    First scale to the specified size in equal proportion to the short edge,
    then center cropping
    """

    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, list):
            size = tuple(size)
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(
            clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode
        )
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class TemporalRandomCrop:
    """Temporally crop the given frame indices at a random location.
    Args:
            size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index


class Expand2Square:
    """
    Expand the given PIL image to a square by padding it with a background color.
    Args:
        mean (sequence): Sequence of means for each channel.
    """

    def __init__(self, mean):
        self.background_color = tuple(int(x * 255) for x in mean)

    def __call__(self, pil_img):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), self.background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), self.background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class JpegDegradationSimulator:
    """
    Degrade an image based on the JPEG quality.
    """

    def __init__(self, min_quality=75, max_quality=101):
        """
        Initialize the simulator with a list of qualities.
        """
        self.qualities = list(range(min_quality, max_quality))
        self.jpeg_degrade_functions = {
            quality: self._simulate_jpeg_degradation(quality) for quality in self.qualities
        }

    def _simulate_jpeg_degradation(self, quality):
        """
        Create a function to degrade an image based on the JPEG quality.
        """

        def jpeg_degrade(img):
            with io.BytesIO() as output:
                img.convert("RGB").save(output, format="JPEG", quality=quality)
                output.seek(0)  # Move the reading cursor to the start of the stream
                img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
            return img_jpeg

        return jpeg_degrade

    def __call__(self, img):
        """
        Apply a random JPEG degradation from the available qualities.
        """
        transform = T.RandomChoice([T.Lambda(self.jpeg_degrade_functions[quality]) for quality in self.qualities])
        return transform(img)

    def __repr__(self):
        """
        Represent the class instance.
        """
        return f"JpegDegradationSimulator(qualities={self.qualities})"


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "intepolate",
        ]
        if not all(
                mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ):
            raise Exception(f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}")
        if not all(
                mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ):
            raise Exception(f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}")
        if not all(
                mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ):
            raise Exception(f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}")
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        if not math.isclose(
                sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ):
            raise Exception(f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}")
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        vae_micro_frame_size = 17
        video_frames = x.shape[1]
        temporal_vae_downsample = 4
        num_frames = (video_frames // vae_micro_frame_size) * \
                     math.ceil(vae_micro_frame_size / temporal_vae_downsample)
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos: random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos: random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1

        return mask
