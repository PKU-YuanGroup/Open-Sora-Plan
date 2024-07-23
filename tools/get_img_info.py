
import cv2
from tqdm import tqdm
from glob import glob
import json
import os
from multiprocessing import Pool
import argparse

def get_image_size(image_path):
    """
    Given an image path, return its width and height.
    """
    try:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return image_path, height, width
    except Exception as e:
        return image_path, None, None

def process_image_paths(image_paths):
    items = []
    for image_path, height, width in image_paths:
        path = image_path.replace(image_root if image_root.endswith('/') else image_root + '/', '')
        item = dict(path=path, resolution=dict(height=height, width=width))
        items.append(item)
    return items

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    args = parser.parse_args()
    # image_root = '/storage/dataset/xx/xx_v1'
    # save_root = '/storage/dataset/image'
    image_root = args.image_root
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    save_name = '{}_{}_resolution.json'
    all_paths = glob(os.path.join(image_root, '**', '*.jpg'), recursive=True)

    num_processes = os.cpu_count()  # Use the number of CPU cores
    num_processes = 128  # Use the number of CPU cores
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(get_image_size, all_paths), total=len(all_paths)))

    items = process_image_paths(results)

    with open(os.path.join(save_root, save_name.format(args.save_name, len(items))), 'w') as f:
        json.dump(items, f, indent=2)
