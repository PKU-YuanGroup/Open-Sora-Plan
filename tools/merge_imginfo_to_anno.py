
from tqdm import tqdm
from glob import glob
import json
import os

# anno_path = '/storage/anno_jsons/human_images_162094.json'
# img_info_path = '/storage/dataset/image/human_images_162094_resolution.json'
# save_root = '/storage/anno_jsons'
# save_name = 'human_images_{}_resolution.json'


# anno_path = '/storage/anno_jsons/tuzhan_mj_4615265.json'
# img_info_path = '/storage/dataset/image/tuzhan_mj_4615530_resolution.json'
# save_root = '/storage/anno_jsons'
# save_name = 'tuzhan_mj_{}_resolution.json'


# anno_path = '/storage/anno_jsons/sam_image_11185255.json'
# img_info_path = '/storage/dataset/image/sam_image_11185362_resolution.json'
# save_root = '/storage/anno_jsons'
# save_name = 'sam_image_{}_resolution.json'


anno_path = '/storage/anno_jsons/civitai_v1_1940032.json'
img_info_path = '/storage/dataset/image/civitai_2151074_resolution.json'
save_root = '/storage/anno_jsons'
save_name = 'civitai_v1_{}_resolution.json'

# anno_path = '/storage/anno_jsons/ideogram_v1_71637.json'
# img_info_path = '/storage/dataset/image/ideogram_v1_71637_resolution.json'
# save_root = '/storage/anno_jsons'
# save_name = 'ideogram_v1_{}_resolution.json'

with open(anno_path, 'r') as f:
    anno = json.load(f)
with open(img_info_path, 'r') as f:
    img_info = json.load(f)
img_info = {i['path']: i['resolution'] for i in img_info}

items = []
cnt = 0
for i in tqdm(anno):
    resolution = img_info[i['path']]
    i['resolution'] = resolution
    items.append(i)
with open(os.path.join(save_root, save_name.format(len(items))), 'w') as f:
    json.dump(items, f, indent=2)