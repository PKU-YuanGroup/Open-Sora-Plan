# import cv2
# from tqdm import tqdm
# from glob import glob
# import json
# import os

# def get_image_size(image_path):
#     """
#     Given an image path, return its width and height.
#     """
#     try:
#         image = cv2.imread(image_path)
#         height, width = image.shape[:2]
#         return height, width
#     except Exception as e:
#         return None, None

# image_root = '/storage/dataset/image/human_images/'
# save_root = '/storage/dataset/image'
# os.makedirs(save_root, exist_ok=True)
# save_name = 'human_images_{}_resolution.json'
# all_paths = glob(os.path.join(image_root, '**', f'*.jpg'), recursive=True)
# items = []
# for i in tqdm(all_paths):
#     height, width = get_image_size(i)
#     path = i.replace(image_root if image_root.endswith('/') else image_root + '/', '')
#     item = dict(path=path, resolution=dict(height=height, width=width))
#     items.append(item)
# with open(os.path.join(save_root, save_name.format(len(items))), 'w') as f:
#     json.dump(items, f, indent=2)




import cv2
from tqdm import tqdm
from glob import glob
import json
import os
from multiprocessing import Pool

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
    image_root = '/storage/dataset/image/tuzhan_mj'
    save_root = '/storage/dataset/image'
    os.makedirs(save_root, exist_ok=True)
    save_name = 'tuzhan_mj_{}_resolution.json'
    all_paths = glob(os.path.join(image_root, '**', '*.jpg'), recursive=True)

    num_processes = os.cpu_count()  # Use the number of CPU cores
    num_processes = 128  # Use the number of CPU cores
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(get_image_size, all_paths), total=len(all_paths)))

    items = process_image_paths(results)

    with open(os.path.join(save_root, save_name.format(len(items))), 'w') as f:
        json.dump(items, f, indent=2)
