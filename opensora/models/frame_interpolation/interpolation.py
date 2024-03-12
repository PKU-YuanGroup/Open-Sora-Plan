# this script is modified from https://github.com/MCG-NKU/AMT/blob/main/demos/demo_2x.py
from json import load
import os
import cv2
import sys
import glob
import torch
import argparse
import numpy as np
import os.path as osp
from warnings import warn
from omegaconf import OmegaConf
from torchvision.utils import make_grid
sys.path.append('.')
from utils.utils import (
    read, write,
    img2tensor, tensor2img,
    check_dim_and_resize
    )
from utils.build_utils import build_from_cfg
from utils.utils import InputPadder


AMT_G = {
    'name': 'networks.AMT-G.Model',
    'params':{
        'corr_radius': 3,
        'corr_lvls': 4,
        'num_flows': 5,
    }
}



def init(device="cuda"):

    '''
        initialize the device and the anchor resolution.
    '''

    if device == 'cuda':
        anchor_resolution = 1024 * 512
        anchor_memory = 1500 * 1024**2
        anchor_memory_bias = 2500 * 1024**2
        vram_avail = torch.cuda.get_device_properties(device).total_memory
        print("VRAM available: {:.1f} MB".format(vram_avail / 1024 ** 2))
    else:
        # Do not resize in cpu mode
        anchor_resolution = 8192*8192
        anchor_memory = 1
        anchor_memory_bias = 0
        vram_avail = 1
    
    return anchor_resolution, anchor_memory, anchor_memory_bias, vram_avail

def get_input_video_from_path(input_path, device="cuda"):

    '''
        Get the input video from the input_path.

        params:
            input_path: str, the path of the input video.
            devices: str, the device to run the model.
        returns:
            inputs: list, the list of the input frames.
            scale: float, the scale of the input frames.
            padder: InputPadder, the padder to pad the input frames.
    '''

    anchor_resolution, anchor_memory, anchor_memory_bias, vram_avail = init(device)

    if osp.splitext(input_path)[-1] in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 
                                        '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.FLV', 
                                        '.WMV', '.WEBM']:

        vcap = cv2.VideoCapture(input_path)

        inputs = []
        w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = anchor_resolution / (h * w) * np.sqrt((vram_avail - anchor_memory_bias) / anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        if scale < 1:
            print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
        padding = int(16 / scale)
        padder = InputPadder((h, w), padding)
        while True:
            ret, frame = vcap.read()
            if ret is False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_t = img2tensor(frame).to(device)
            frame_t = padder.pad(frame_t)
            inputs.append(frame_t)
        print(f'Loading the [video] from {input_path}, the number of frames [{len(inputs)}]')
    else:
        raise TypeError("Input should be a video.")
    
    return inputs, scale, padder


def load_model(ckpt_path, device="cuda"):

    '''
        load the frame interpolation model.
    '''
    network_cfg = AMT_G
    network_name = network_cfg['name']
    print(f'Loading [{network_name}] from [{ckpt_path}]...')
    model = build_from_cfg(network_cfg)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    return model

def interpolater(model, inputs, scale, padder, iters=1):

    '''
        interpolating with the interpolation model.

        params:
            model: nn.Module, the frame interpolation model.
            inputs: list, the list of the input frames.
            scale: float, the scale of the input frames.
            iters: int, the number of iterations of interpolation. The final frames model generating is 2 ** iters * (m - 1) + 1 and m is input frames. 
        returns:
            outputs: list, the list of the output frames.
    '''

    print(f'Start frame interpolation:')
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    for i in range(iters):
        print(f'Iter {i+1}. input_frames={len(inputs)} output_frames={2*len(inputs)-1}')
        outputs = [inputs[0]]
        for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
            in_0 = in_0.to(device)
            in_1 = in_1.to(device)
            with torch.no_grad():
                imgt_pred = model(in_0, in_1, embt, scale_factor=scale, eval=True)['imgt_pred']
            outputs += [imgt_pred.cpu(), in_1.cpu()]
        inputs = outputs

    outputs = padder.unpad(*outputs)

    return outputs

def write(outputs, input_path, output_path, frame_rate=30):
    '''
        write results to the output_path.
    '''

    if osp.exists(output_path) is False:
        os.makedirs(output_path)

    
    size = outputs[0].shape[2:][::-1]

    _, file_name_with_extension = os.path.split(input_path)
    file_name, _ = os.path.splitext(file_name_with_extension)

    save_video_path = f'{output_path}/output_{file_name}.mp4'
    writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 
                        frame_rate, size)

    for i, imgt_pred in enumerate(outputs):
        imgt_pred = tensor2img(imgt_pred)
        imgt_pred = cv2.cvtColor(imgt_pred, cv2.COLOR_RGB2BGR)
        writer.write(imgt_pred)        
    print(f"Demo video is saved to [{save_video_path}]")

    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='amt-g.pth', help="The pretrained model.") 
    parser.add_argument('--niters', type=int, default=1, help="Iter of Interpolation. The number of frames will be double after per iter.") 
    parser.add_argument('--input', default="test.mp4", help="Input video.") 
    parser.add_argument('--output_path', type=str, default='results', help="Output path.") 
    parser.add_argument('--frame_rate', type=int, default=30, help="Frames rate of the output video.")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt
    input_path = args.input
    output_path = args.output_path
    iters = int(args.niters)
    frame_rate = int(args.frame_rate)

    inputs, scale, padder = get_input_video_from_path(input_path, device)
    model = load_model(ckpt_path, device)
    outputs = interpolater(model, inputs, scale, padder, iters)
    write(outputs, input_path, output_path, frame_rate)
