import json
import pickle
import random
import os
import re
from pathlib import Path

from tqdm import tqdm

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 删除所有控制字符
    cleaned_content = re.sub(r"[\x00-\x1F]", "", content)

    try:
        data = json.loads(cleaned_content)
        print("解析成功")
        return data
    except json.JSONDecodeError as e:
        print("解析失败:", e)
        return []

def save_data(data, output_dir, k):
    """将数据打乱并均分为 k 份 JSON 文件"""
    random.shuffle(data)
    chunk_size = len(data) // k
    for i in tqdm(range(k), desc='Saving'):
        chunk = data[i * chunk_size: (i + 1) * chunk_size] if i < k - 1 else data[i * chunk_size:]
        output_file = os.path.join(output_dir, f'final_{i + 1}_{len(chunk)}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=4)

def process_files(file_list, root_path_list, output_dir, k):
    """处理文件，修改 path 属性并均分"""
    all_data = []
    
    for file_path, root_path in tqdm(zip(file_list, root_path_list), desc='Processing', total=len(file_list)):
        data = load_data(file_path)
        for item in data:
            if 'path' in item:
                item['path'] = os.path.join(root_path, os.path.basename(item['path']))
        all_data.extend(data)
    
    save_data(all_data, output_dir, k)

# 示例使用
file_list = [
    "/work/share1/video_final/istock_v1_final_321725.json",
    "/work/share1/video_final/istock_v2_final_254054.json",
    "/work/share1/video_final/istock_v4_final_1609738.json",
    "/work/share1/video_final/sucai_final_3880570.json",
    "/work/share1/video_final/xigua_final_2163747.json",
    "/work/share1/video_final/20241001_final_849158.json",
    "/work/share1/video_final/20241022_final_1119970.json",
    "/work/share1/video_final/chongwu_20241006_final_281810.json",
    "/work/share1/video_final/chongwu_20241120_final_495720.json",
    "/work/share1/video_final/tiyu_20241006_final_83899.json",
    "/work/share1/video_final/tiyu_20241120_final_439606.json",
    "/work/share1/video_final/dongchedi_20241005_final_475885.json",
    "/work/share1/video_final/high_1_final_1978974.json",
    "/work/share1/video_final/high_2_final_4679455.json",
    "/work/share1/video_final/high_3_final_6702395.json",
    "/work/share1/video_final/high_4_final_2304907.json"
    "/work/share1/video_final/aiqiyi_final_30345.json",
    "/work/share1/video_final/blbl_final_941569.json",
    "/work/share1/video_final/movie_ctv01_final_1544973.json",
    "/work/share1/video_final/movie_ctv03_final_717788.json",
    "/work/share1/video_final/movie_ctv04_final_711290.json",
    "/work/share1/video_final/movie_ctv05_final_312631.json",
    "/work/share1/video_final/movie_ctv05_2_final_869060.json"
    "/work/share1/video_final/movie_bbc01_final_300528.json",
    "/work/share1/video_final/movie_bbc02_final_385967.json",
    "/work/share1/video_final/movie_bbc03_final_1043580.json",
    "/work/share1/video_final/movie_bbc04_final_413116.json",
    "/work/share1/video_final/movie_bbc05_final_698107.json",
]  # 你的 JSON/PKL 文件列表
root_path_list = [
    # "share/dataset/sucai_video/istock_v1",
    # "share/dataset/sucai_video/istock_v2",
    # "share/dataset/sucai_video/istock_v4",
    # "share/dataset/sucai_video",
    # "share/dataset/xigua_video",
    # "share/dataset/xigua_video",
    # "share/dataset/xigua_video",
    # "share/dataset/xigua_video",
    # "share/dataset/xigua_video",
    # "share/dataset/xigua_video",
    # "share/dataset/xigua_video",
    # "share/dataset/xigua_video",
    "share/dataset/xigua_video",
    "share/dataset/xigua_video",
    "share/dataset/xigua_video",
    "share/dataset/xigua_video",
    "share/dataset/movie_video/aiqiyi",
    "share/dataset/movie_video/bilibili",
    "share/dataset/movie_video/ctv",
    "share/dataset/movie_video/ctv",
    "share/dataset/movie_video/ctv",
    "share/dataset/movie_video/ctv",
    "share/dataset/movie_video/ctv"
    "share1/video1/bbc01",
    "share1/video1/bbc01",
    "share1/video1/bbc01",
    "share1/video1/bbc01",
    "share1/video1/bbc01",
]  # 对应的根路径
output_dir = "/work/share1/caption/osp"  # 输出文件夹
k = 20  # 分割为 k 个 JSON

os.makedirs(output_dir, exist_ok=True)
process_files(file_list, root_path_list, output_dir, k)
