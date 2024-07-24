
from tqdm import tqdm
from glob import glob
import json
import os



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--img_info_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    args = parser.parse_args()

    # anno_path = '/storage/anno_jsons/sam_1940032.json'
    # img_info_path = '/storage/dataset/image/sam_2151074_resolution.json'
    # save_root = '/storage/anno_jsons'
    # save_name = 'sam_{}_resolution.json'
    
    anno_path = args.anno_path
    img_info_path = args.img_info_path
    save_root = args.save_root
    save_name = '{}_{}_resolution.json'


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
    with open(os.path.join(save_root, save_name.format(args.save_name, len(items))), 'w') as f:
        json.dump(items, f, indent=2)