import os.path as osp
import math
import pickle
import warnings
import glob
from PIL import Image
import torch.utils.data as data
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.datasets.video_utils import VideoClips

class CausalVAEDataset(data.Dataset):
    video_exts = ['avi', 'mp4', 'webm']
    def __init__(self, video_folder, sequence_length, sample_rate=1, train=True, resolution=64):
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        video_files = []
        for data_folder in [video_folder]:
            if data_folder is None:
                continue
            folder = data_folder
            video_files += sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                         for ext in self.video_exts], [])
            
        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}_{sample_rate}.pkl")
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
        if not osp.exists(cache_file):
            clips = VideoClips(video_files, sequence_length, frame_rate=sample_rate, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(video_files, sequence_length, frame_rate=sample_rate, 
                               _precomputed_metadata=metadata)
        if dist.is_initialized() and dist.get_rank() == 0:
            dist.barrier()
        self._clips = clips
        self._clips_num = self._clips.num_clips()
        
    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips_num

    def __getitem__(self, idx):
        resolution = self.resolution
        if idx < self._clips_num:
            video, _, _, idx = self._clips.get_clip(idx)
            video = preprocess(video, resolution)
            class_name = get_parent_dir(self._clips.video_paths[idx])
        else:
            idx -= self._clips_num
            image = Image.open(self.image_files[idx])
            video = preprocess_image(image, resolution, self.sequence_length)
        # label = self.class_to_label[class_name]
        return dict(video=video, label="")

def get_parent_dir(path):
    return osp.basename(osp.dirname(path))

def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    # video -= 0.5
    video = 2.0 * video - 1

    return video
