import glob
import argparse
import h5py
import numpy as np
from PIL import Image
import os
import os.path as osp
from tqdm import tqdm
import sys

def convert_data(f, split):
    root_dir = args.data_dir
    path = osp.join(root_dir, 'processed_data', split)
    traj_paths = glob.glob(osp.join(path, '*', '*'))
    trajs, actions = [], []
    for traj_path in tqdm(traj_paths):
        image_paths = glob.glob(osp.join(traj_path, '*.png'))
        image_paths.sort(key=lambda x: int(osp.splitext(osp.basename(x))[0]))
        traj = []
        for img_path in image_paths:
            img = Image.open(img_path)
            arr = np.array(img) # HWC
            traj.append(arr)
        traj = np.stack(traj, axis=0) # THWC
        trajs.append(traj)

        actions.append(np.load(osp.join(traj_path, 'actions.npy')))

    idxs = np.arange(len(trajs)) * 30
    trajs = np.concatenate(trajs, axis=0) # (NT)HWC
    actions = np.concatenate(actions, axis=0) # (NT)(act_dim)

    f.create_dataset(f'{split}_data', data=trajs)
    f.create_dataset(f'{split}_actions', data=actions)
    f.create_dataset(f'{split}_idx', data=idxs)

    print(split)
    print(f'\timages: {f[f"{split}_data"].shape}, {f[f"{split}_data"].dtype}')
    print(f'\timages: {f[f"{split}_idx"].shape}, {f[f"{split}_idx"].dtype}')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
f = h5py.File(osp.join(args.output_dir, 'bair.hdf5'), 'a')
convert_data(f, 'train')
convert_data(f, 'test')
f.close()
