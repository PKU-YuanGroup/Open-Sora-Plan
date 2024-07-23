import os

import torch

import os
import math
import torch
import logging
import random
import subprocess
import numpy as np
import torch.distributed as dist

# from torch._six import inf
from torch import inf
from PIL import Image
from typing import Union, Iterable
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from diffusers.utils import is_bs4_available, is_ftfy_available

import html
import re
import urllib.parse as ul

from torch.utils.data import Dataset
from torchvision.transforms import Lambda, Compose
from decord import VideoReader, cpu

from torchvision.transforms._transforms_video import CenterCropVideo
from torch.nn import functional as F
import cv2
import numpy.typing as npt

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def find_model(model_name):
    """
    Finds a pre-trained Latte model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find Latte checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)

    # if "ema" in checkpoint:  # supports checkpoints from train.py
    #     print('Using Ema!')
    #     checkpoint = checkpoint["ema"]
    # else:
    print('Using model!')
    checkpoint = checkpoint['model']
    return checkpoint

#################################################################################
#                             Training Clip Gradients                           #
#################################################################################
import deepspeed
def print_grad_norm(model):
    # 计算并打印梯度范数
    # model_engine = accelerator.deepspeed_engine_wrapped.engine
    # gradients = model_engine.get_gradients()
    # grad_norm = get_grad_norm(gradients)
    # 计算并打印梯度范数
    grad_norm = 0
    n_grad = 0
    for name, param in model.named_parameters():
        grad_data = deepspeed.utils.safe_get_full_grad(param)
        # self.print_tensor_stats(grad_data, name=name)

        if grad_data is not None:
            param_norm = grad_data.norm(2)
            grad_norm += param_norm.item() ** 2
            n_grad += 1
    grad_norm = (grad_norm / n_grad) ** (1. / 2)

    # self.print_msg('=' * 50)
    print(f'Gradient Norm is : {grad_norm}')

def get_grad_norm(
        parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm


def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad=True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        # gradient_cliped = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        # print(gradient_cliped)
    return total_norm


def get_experiment_dir(root_dir, args):
    # if args.pretrained is not None and 'Latte-XL-2-256x256.pt' not in args.pretrained:
    #     root_dir += '-WOPRE'
    if args.use_compile:
        root_dir += '-Compile'  # speedup by torch compile
    if args.attention_mode:
        root_dir += f'-{args.attention_mode.upper()}'
    # if args.enable_xformers_memory_efficient_attention:
    #     root_dir += '-Xfor'
    if args.gradient_checkpointing:
        root_dir += '-Gc'
    if args.mixed_precision:
        root_dir += f'-{args.mixed_precision.upper()}'
    root_dir += f'-{args.max_image_size}'
    return root_dir

def get_precision(args):
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dtype

#################################################################################
#                             Training Logger                                   #
#################################################################################

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)

    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard(tensorboard_dir):
    """
    Create a tensorboard that saves losses.
    """
    if dist.get_rank() == 0:  # real tensorboard
        # tensorboard
        writer = SummaryWriter(tensorboard_dir)

    return writer


def write_tensorboard(writer, *args):
    '''
    write the loss information to a tensorboard file.
    Only for pytorch DDP mode.
    '''
    if dist.get_rank() == 0:  # real tensorboard
        writer.add_scalar(args[0], args[1], args[2])


#################################################################################
#                      EMA Update/ DDP Training Utils                           #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = "29566"
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


#################################################################################
#                             Testing  Utils                                    #
#################################################################################

def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros((t, (padding + h) * nrow + padding,
                              (padding + w) * ncol + padding, c), dtype=torch.uint8)

    print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    return video_grid


#################################################################################
#                             MMCV  Utils                                    #
#################################################################################


def collect_env():
    # Copyright (c) OpenMMLab. All rights reserved.
    from mmcv.utils import collect_env as collect_base_env
    from mmcv.utils import get_git_hash
    """Collect the information of the running environments."""

    env_info = collect_base_env()
    env_info['MMClassification'] = get_git_hash()[:7]

    for name, val in env_info.items():
        print(f'{name}: {val}')

    print(torch.cuda.get_arch_list())
    print(torch.version.cuda)


#################################################################################
#                          Pixart-alpha  Utils                                  #
#################################################################################


bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')  # noqa

def text_preprocessing(text):
    # The exact text cleaning as was in the training stage:
    text = clean_caption(text)
    text = clean_caption(text)
    return text

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def clean_caption(caption):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub('<person>', 'person', caption)
    # urls:
    caption = re.sub(
        r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '', caption)  # regex for urls
    caption = re.sub(
        r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '', caption)  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features='html.parser').text

    # @<nickname>
    caption = re.sub(r'@[\w\d]+\b', '', caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
    caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
    caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
    caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
    caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
    caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
    caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
        '-', caption)

    # кавычки к одному стандарту
    caption = re.sub(r'[`´«»“”¨]', '"', caption)
    caption = re.sub(r'[‘’]', "'", caption)

    # &quot;
    caption = re.sub(r'&quot;?', '', caption)
    # &amp
    caption = re.sub(r'&amp', '', caption)

    # ip adresses:
    caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

    # article ids:
    caption = re.sub(r'\d:\d\d\s+$', '', caption)

    # \n
    caption = re.sub(r'\\n', ' ', caption)

    # "#123"
    caption = re.sub(r'#\d{1,3}\b', '', caption)
    # "#12345.."
    caption = re.sub(r'#\d{5,}\b', '', caption)
    # "123456.."
    caption = re.sub(r'\b\d{6,}\b', '', caption)
    # filenames:
    caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

    #
    caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

    caption = re.sub(bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r'(?:\-|\_)')
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, ' ', caption)

    caption = basic_clean(caption)

    caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
    caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
    caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

    caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
    caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
    caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
    caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
    caption = re.sub(r'\bpage\s+\d+\b', '', caption)

    caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

    caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

    caption = re.sub(r'\b\s+\:\s+', r': ', caption)
    caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
    caption = re.sub(r'\s+', ' ', caption)

    caption.strip()

    caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
    caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
    caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
    caption = re.sub(r'^\.\S+$', '', caption)

    return caption.strip()


#################################################################################
#                          eval PSNR when training                                  #
#################################################################################

def resize(x, resolution):
    height, width = x.shape[-2:]
    aspect_ratio = width / height
    if width <= height:
        new_width = resolution
        new_height = int(resolution / aspect_ratio)
    else:
        new_height = resolution
        new_width = int(resolution * aspect_ratio)
    resized_x = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True, antialias=True)
    return resized_x

def _preprocess(video_data, short_size=128, crop_size=None):
    transform = Compose(
        [
            Lambda(lambda x: (x / 255.0)),
            Lambda(lambda x: resize(x, short_size)),
            (
                CenterCropVideo(crop_size=crop_size)
                if crop_size is not None
                else Lambda(lambda x: x)
            ),
        ]
    )
    video_outputs = transform(video_data)
    video_outputs = _format_video_shape(video_outputs)
    return video_outputs

def _format_video_shape(video, time_compress=4, spatial_compress=8):
    time = video.shape[1]
    height = video.shape[2]
    width = video.shape[3]
    new_time = (
        (time - (time - 1) % time_compress)
        if (time - 1) % time_compress != 0
        else time
    )
    new_height = (
        (height - (height) % spatial_compress)
        if height % spatial_compress != 0
        else height
    )
    new_width = (
        (width - (width) % spatial_compress) if width % spatial_compress != 0 else width
    )
    return video[:, :new_time, :new_height, :new_width]

def array_to_video(
    image_array: npt.NDArray, fps: float = 30.0, output_file: str = "output_video.mp4"
) -> None:
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()


def custom_to_video(
    x: torch.Tensor, fps: float = 2.0, output_file: str = "output_video.mp4"
) -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 2, 3, 0).float().numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return

class RealVideoDataset(Dataset):
    def __init__(
        self,
        real_video_dir,
        num_frames,
        sample_rate=1,
        crop_size=None,
        resolution=128,
    ) -> None:
        super().__init__()
        self.real_video_files = self._combine_without_prefix(real_video_dir)
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution

    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_video_file = self.real_video_files[index]
        real_video_tensor = self._load_video(real_video_file)
        video_name = os.path.basename(real_video_file)
        return {'video': real_video_tensor, 'file_name': video_name }

    def _load_video(self, video_path):
        num_frames = self.num_frames
        sample_rate = self.sample_rate
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames > sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            s = 0
            e = total_frames
            num_frames = int(total_frames / sample_frames_len * num_frames)
            print(
                f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
                video_path,
                total_frames,
            )

        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(3, 0, 1, 2)
        return _preprocess(
            video_data, short_size=self.short_size, crop_size=self.crop_size
        )

    def _combine_without_prefix(self, folder_path, prefix="."):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(os.path.join(folder_path, name))
        folder.sort()
        return folder