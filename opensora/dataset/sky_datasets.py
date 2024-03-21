import os
import torch
import random
import torch.utils.data as data

import numpy as np

from PIL import Image

from opensora.utils.dataset_utils import is_image_file


class Sky(data.Dataset):
    def __init__(self, args, transform, temporal_sample=None, train=True):

        self.args = args
        self.data_path = args.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.num_frames = self.args.num_frames
        self.sample_rate = self.args.sample_rate
        self.data_all = self.load_video_frames(self.data_path)
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        if self.use_image_num != 0 and not self.use_img_from_vid:
            self.img_cap_list = self.get_img_cap_list()

    def __getitem__(self, index):

        vframes = self.data_all[index]
        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, num=self.num_frames, dtype=int) # start, stop, num=50

        select_video_frames = vframes[frame_indice[0]: frame_indice[-1]+1: self.sample_rate]

        video_frames = []
        for path in select_video_frames:
            video_frame = torch.as_tensor(np.array(Image.open(path), dtype=np.uint8, copy=True)).unsqueeze(0)
            video_frames.append(video_frame)
        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)
        video_clip = video_clip.transpose(0, 1)  # T C H W -> C T H W

        if self.use_image_num != 0 and self.use_img_from_vid:
            select_image_idx = np.linspace(0, self.num_frames - 1, self.use_image_num, dtype=int)
            assert self.num_frames >= self.use_image_num
            images = video_clip[:, select_image_idx]  # c, num_img, h, w
            video_clip = torch.cat([video_clip, images], dim=1)  # c, num_frame+num_img, h, w
        elif self.use_image_num != 0 and not self.use_img_from_vid:
            images, captions = self.img_cap_list[index]
            raise NotImplementedError
        else:
            pass

        return video_clip, 1

    def __len__(self):
        return self.video_num
    
    def load_video_frames(self, dataroot):
        data_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = [i for i in meta[2] if is_image_file(i)]
                frames = sorted(frames, key=lambda item: int(item.split('.')[0].split('_')[-1]))
            except:
                pass
                # print(meta[0]) # root
                # print(meta[2]) # files
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            if len(frames) > max(0, self.num_frames * self.sample_rate): # need all > (16 * frame-interval) videos
            # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                data_all.append(frames)
        self.video_num = len(data_all)
        return data_all

    def get_img_cap_list(self):
        raise NotImplementedError

if __name__ == '__main__':

    import argparse
    import torchvision
    import video_transforms
    import torch.utils.data as data

    from torchvision import transforms
    from torchvision.utils import save_image


    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=4)
    parser.add_argument("--data-path", type=str, default="/path/to/datasets/sky_timelapse/sky_train/")
    config = parser.parse_args()


    target_video_len = config.num_frames

    temporal_sample = video_transforms.TemporalRandomCrop(target_video_len * config.frame_interval)
    trans = transforms.Compose([
        video_transforms.ToTensorVideo(),
        # video_transforms.CenterCropVideo(256),
        video_transforms.CenterCropResizeVideo(256),
        # video_transforms.RandomHorizontalFlipVideo(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    taichi_dataset = Sky(config, transform=trans, temporal_sample=temporal_sample)
    print(len(taichi_dataset))
    taichi_dataloader = data.DataLoader(dataset=taichi_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, video_data in enumerate(taichi_dataloader):
        print(video_data['video'].shape)
        
        # print(video_data.dtype)
        # for i in range(target_video_len):
        #     save_image(video_data[0][i], os.path.join('./test_data', '%04d.png' % i), normalize=True, value_range=(-1, 1))

        # video_ = ((video_data[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        # torchvision.io.write_video('./test_data' + 'test.mp4', video_, fps=8)
        # exit()