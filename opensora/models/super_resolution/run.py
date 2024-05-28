import cv2
import argparse
from basicsr.test_img import image_sr  
from os import path as osp
import os
import shutil
from PIL import Image
import re
import imageio.v2 as imageio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
# from natsort import natsortedf

def replace_filename(original_path, suffix):
    directory = os.path.dirname(original_path)
    old_filename = os.path.basename(original_path)
    name_part, file_extension = os.path.splitext(old_filename)
    new_filename = f"{name_part}{suffix}{file_extension}"
    new_path = os.path.join(directory, new_filename)

    return new_path

def create_temp_folder(folder_path):
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def delete_temp_folder(folder_path):
    shutil.rmtree(folder_path)

def extract_number(filename):
    s = re.findall(r'\d+', filename)
    return int(s[0]) if s else -1

def bicubic_upsample_opencv(input_image_path, output_image_path, scale_factor):
    
    img = cv2.imread(input_image_path)
    
    original_height, original_width = img.shape[:2]
    
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    upsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_image_path, upsampled_img)


def process_frame(frame_count, frame, temp_LR_folder_path, temp_HR_folder_path, SR):
    frame_path = os.path.join(temp_LR_folder_path, f"frame_{frame_count}{SR}.png")
    cv2.imwrite(frame_path, frame)
    HR_frame_path = os.path.join(temp_HR_folder_path, f"frame_{frame_count}.png")

    if SR == 'x4': 
        bicubic_upsample_opencv(frame_path, HR_frame_path, 4)
    elif SR == 'x2': 
        bicubic_upsample_opencv(frame_path, HR_frame_path, 2)


def images_to_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.replace("frame_","").split('.')[0])) 
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def video_sr(args):
    file_name = os.path.basename(args.input_dir)
    video_output_path = os.path.join(args.output_dir,file_name)
    
    directory, filename = os.path.split(args.input_dir)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_HR{ext}"

    if args.SR == 'x4': 
        temp_LR_folder_path = os.path.join(args.output_dir, f'temp_LR/X4')
        video_output_path = replace_filename(video_output_path, '_x4')
        new_filename = f"{name}_HR_x4{ext}"
       
    if args.SR == 'x2': 
        temp_LR_folder_path = os.path.join(args.output_dir, f'temp_LR/X2')
        video_output_path = replace_filename(video_output_path, '_x2')
        new_filename = f"{name}_HR_x2{ext}"
        
    result_out = os.path.join(directory, new_filename)
    temp_HR_folder_path = os.path.join(args.output_dir, f'temp_HR')
    
    # create_temp_folder(result_temp)
    create_temp_folder(temp_LR_folder_path)
    create_temp_folder(temp_HR_folder_path) 

    cap = cv2.VideoCapture(args.input_dir)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    t1 = time.time()
    frame_count = 0
    frames_to_process = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_to_process.append((frame_count, frame))
        frame_count += 1

    with ThreadPoolExecutor(max_workers = args.mul_numwork) as executor:
        for frame_count, frame in frames_to_process:
            executor.submit(process_frame, frame_count, frame, temp_LR_folder_path, temp_HR_folder_path, args.SR)
    
    print("total frames:",frame_count)
    print("fps :",cap.get(cv2.CAP_PROP_FPS))
    
    t2 = time.time()
    print('get frames from video: ',t2 - t1,'s')
    # progress all frames in video
    image_sr(args)

    t3 = time.time()
    print('image super resolution: ',t3 - t2,'s')
    # recover video form all frames
    images_to_video(temp_HR_folder_path, result_out, cap.get(cv2.CAP_PROP_FPS))

    t4 = time.time()
    print('tranformer frames to  video: ',t4 - t3,'s')
    # release all resources
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGT for Video Super-Resolution")
    # make sure you SR is match with the ckpt_path
    parser.add_argument("--SR", type=str, choices=['x2', 'x4'], default='x2', help='image resolution')
    parser.add_argument("--ckpt_path", type=str, default = "/remote-home/lzy/super_video/RGT/experiments/pretrained_models/RGT_x2.pth")
    
    parser.add_argument("--root_path", type=str, default = "/remote-home/lzy/super_video/RGT")
    parser.add_argument("--input_dir", type=str, default= "/remote-home/lzy/super_video/RGT/input_video/test2.mp4")
    parser.add_argument("--output_dir", type=str, default= "/remote-home/lzy/super_video/RGT/input_video/output_10s_2")
    
    parser.add_argument("--mul_numwork", type=int, default = 16, help ='max_workers to execute Multi')
    parser.add_argument("--use_chop", type= bool, default = True,  help ='use_chop: True  # True to save memory, if img too large')
    args = parser.parse_args()
    video_sr(args)
