import sys
import os
import os.path as osp
import shutil

root = sys.argv[1]
fold = int(sys.argv[2])
assert fold in [1, 2, 3]

def move_files(files, split):
    split_dir = osp.join(root, split)
    os.makedirs(split_dir, exist_ok=True)
    for filename in files:
        folder = osp.join(split_dir, osp.dirname(filename))
        os.makedirs(folder, exist_ok=True)
        shutil.move(osp.join(root, 'UCF-101', filename), osp.join(split_dir, filename))

with open(osp.join(root, 'ucfTrainTestlist', f'trainlist0{fold}.txt'), 'r') as f:
    train_files = [p.strip().split(' ')[0] for p in f.readlines()]
move_files(train_files, 'train')

with open(osp.join(root, 'ucfTrainTestlist', f'testlist0{fold}.txt'), 'r') as f:
    test_files = [p.strip() for p in f.readlines()]
move_files(test_files, 'test')
