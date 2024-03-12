import os
import pickle
import argparse

from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sample_rate", type=int, default=3)
    parser.add_argument("--data-root", type=str, default="datasets/sky_timelapse/sky_train")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root

    target_video_len = args.num_frames
    frame_interval = args.sample_rate

    data_all = []
    for vid_folder in tqdm(os.listdir(data_root)):
        vid_folder = os.path.join(data_root, vid_folder)
        if not os.path.isdir(vid_folder):
            continue
        sub_vid_folders = [os.path.join(vid_folder, x) for x in os.listdir(vid_folder)]

        # split image files and sub folders
        sub_vid_files = [x for x in sub_vid_folders if not os.path.isdir(x)]
        sub_vid_folders = [x for x in sub_vid_folders if os.path.isdir(x)]

        if len(sub_vid_files) > 0:
            frames = [x for x in sub_vid_files if is_image_file(x)]
            frames = sorted(frames, key=lambda item: int(item.split('.')[0].split('_')[-1]))
            if len(frames) > max(0, target_video_len * frame_interval): # need all > (16 * frame-interval) videos
                # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                data_all.append(frames)

        for sub_vid_folder in sub_vid_folders:
            frames = [os.path.join(sub_vid_folder, x) for x in os.listdir(sub_vid_folder) if is_image_file(x)]
            frames = sorted(frames, key=lambda item: int(item.split('.')[0].split('_')[-1]))
            if len(frames) > max(0, target_video_len * frame_interval): # need all > (16 * frame-interval) videos
                # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                data_all.append(frames)

    with open(os.path.join(data_root, f'metadata_{target_video_len}_{frame_interval}.pkl'), 'wb') as f:
        pickle.dump(data_all, f)


if __name__ == '__main__':
    main()
