import random
from opensora.dataset.inpaint_utils import get_mask_tensor,MaskType
import torch
from opensora.dataset.transform import ToTensorVideo
from opensora.models.causalvideovae import ae_norm
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
from enum import Enum, auto
from ultralytics import YOLO


import torch_npu
from torch_npu.contrib import transfer_to_npu


os.environ['YOLO_VERBOSS'] = 'False'

class MaskType(Enum):
    Semantic_mask = auto()
    bbox_mask = auto()
    background_mask = auto()
    fixed_mask = auto()
    Semantic_expansion_mask = auto()
    fixed_bg_mask = auto()
    t2iv_mask = auto()
    i2v_mask = auto()
    transition_mask = auto()
    v2v_mask = auto()
    clear_mask = auto()
    random_mask = auto()




class single_info:
    def __init__(self, id, label, shape) -> None:
        self.id = id
        self.label = label
        self.shape = shape
        self.frame_indexes = []
        self.infos = []
    def update(self,frame_index,box,conf,mask):
        self.frame_indexes.append(frame_index)
        info = dict(
            box=box,
            conf=conf,
            mask=mask,
        )
        self.infos.append(info)
    def return_dict(self,):
        return dict(
            id=self.id,
            label=self.label,
            frame_size=self.shape,
            frame_index_list = self.frame_indexes,
            infos_list = self.infos
        )

def save_videos_from_pil(pil_images, path, fps=24):
    """
    pil_images: list[Image,...]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    image  = pil_images[0]

    image = ndarray_to_pil(pil_images[0])
    width, height = image.size
    

    codec = "libx264"
    container = av.open(path, "w")
    stream = container.add_stream(codec, rate=fps)

    stream.width = width
    stream.height = height

    for pil_image in pil_images:
        # pil_image = Image.fromarray(image_arr).convert("RGB")
        pil_image = ndarray_to_pil(pil_image)
        av_frame = av.VideoFrame.from_image(pil_image)
        container.mux(stream.encode(av_frame))
    container.mux(stream.encode())
    container.close()

def read_frames(video_tensor) -> list:
    """
    读取视频，返回一个元素类型为ndarray的列表
    """
    # container = av.open(video_path)
    T = video_tensor.shape[0]
    frames = []
    for t in range(T):
        frame_tensor = video_tensor[t]
        frame_tensor = frame_tensor.cpu().numpy()
        frame_tensor = np.transpose(frame_tensor, (1, 2, 0))
        frames.append(frame_tensor)
    return frames


def get_masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
    masked_img = image * (1-mask)
    return masked_img # shape: [H,W,C]; range: [0, 255]

def get_bbox_image(image: np.ndarray,bbox,obj_id):
        # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        bbox_image = image.copy()
        bbox_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 0
        # cv2.putText(image, f'ID: {obj_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        return bbox_image



def select_bg_from_video(bg_masks, video):
    new_container = []
    for index, frame in enumerate(video):
        
        mask = bg_masks[index]
        masked_frame = get_masked_image(frame, mask)
        new_container.append(masked_frame)
    return new_container  

def get_random_box(image_tensor):

    H, W,C = image_tensor.shape

    box_min_size = min(H,W)/2
    box_max_size = max(H,W)/2

    # 随机确定 box 的宽高
    box_width = random.randint(box_min_size, min(box_max_size, W))
    box_height = random.randint(box_min_size, min(box_max_size, H))

    # 随机确定 box 的左上角坐标
    x_start = random.randint(0, W - box_width)
    y_start = random.randint(0, H - box_height)

    box = (x_start, y_start, x_start + box_width, y_start + box_height)

    return box

def combine_masks_and_get_background(masks):
    """
    合并所有 mask 并取反得到背景 mask
    """
    combined_mask = np.any(masks, axis=0)
    background_mask = np.logical_not(combined_mask)
    return background_mask

def parser_results_for_ids(results, frame_size=None):
    id_record = []
    single_info_ins = {}
    background_masks = []
    for frame_index, result in enumerate(results):
        result = result[0]
        if frame_index == 0 and frame_size is None:
            frame_size = result.boxes.orig_shape
        id = result.boxes.id

        # 如果没有检测到物体
        if id is None:
            background_masks.append(np.ones((frame_size)) * 255)
            continue

        id = id.tolist()
        cls = result.boxes.cls.tolist() #每个id对应的label
        conf = result.boxes.conf.tolist() #每个id对应的预测置信度
        box_n = result.boxes.xyxy.tolist() #每个id对应的box
        mask = result.masks.data.cpu().detach().numpy() #每个id对应的mask
        background_masks.append(combine_masks_and_get_background(mask))

        for i, iden in enumerate(id):
            if iden not in id_record:
                id_record.append(iden)
                single_info_ins[iden] = single_info(iden, cls[i], frame_size)
            single_info_ins[iden].update(frame_index, box_n[i], conf[i],mask[i])
    return_list = []
    for _, value in single_info_ins.items():
        return_list.append(value.return_dict())
    return return_list, background_masks
    

def get_mask(video_tensor,mask_type,yole_model):

    video = read_frames(video_tensor=video_tensor)

    # video_tensor_batch = video_tensor.unsqueeze(1)
    T,C,H,W = video_tensor.shape

    # video = video_tensor

    tracker = yole_model

    results = []


    for t in range(T):
        frame_tensor = video_tensor[t]  # 获取当前帧, (C, H, W)
        frame_tensor = frame_tensor.data.cpu().numpy()  # 转为numpy
        frame_tensor = np.transpose(frame_tensor, (1, 2, 0))

        # 进行推理
        result = tracker.track(frame_tensor,save=False, retina_masks=True, agnostic_nms=True,half=True,verbose=False,nms=False)
        
        # 保存结果
        results.append(result)


    parser_res, background_masks = parser_results_for_ids(results)

    select_index = -1
    object_info = []
    frame_indexes = []
    infos = []


    #随机选择一个被追踪物体
    if len(parser_res) is not 0:
        select_index = random.randint(0, len(parser_res)-1)
        object_info = parser_res[select_index]
        frame_indexes = object_info['frame_index_list']
        infos = object_info['infos_list']
        # print("infos size",len(infos))
        # print("frame_indexed",len(frame_indexes))
    else:
        mask_type = MaskType.fixed_mask



    # print("frame_indexed:",frame_indexes)
    # print("infos:",infos)

    # mask_type = MaskType.Semantic_mask

    if mask_type == MaskType.Semantic_mask or mask_type == MaskType.Semantic_expansion_mask:
        Semantic_masks = []
        mask_container = []
        info_index = 0
        for index, frame in enumerate(video):
            if index in frame_indexes:

                mask = infos[info_index]['mask']
                info_index = info_index + 1 

                if mask_type == MaskType.Semantic_expansion_mask:
                    kernel = np.ones((5, 5), np.uint8)
                    # 进行膨胀操作
                    mask = cv2.dilate(mask, kernel, iterations=1)

                # 计算掩码中前景像素的数量
                foreground_pixels = np.sum(mask)

                # 计算图像的总像素数
                total_pixels = mask.size  # 或者使用 image.shape[0] * image.shape[1]

                # 计算比例
                ratio = foreground_pixels / total_pixels
                
                if ratio < 0.2:
                    if random.random() < 0.5:
                        mask_type = MaskType.fixed_mask
                        break
                
                masked_frame = get_masked_image(frame, mask)
                mask_container.append(masked_frame)
                Semantic_masks.append(mask)
            else:
                mask_container.append(np.zeros_like(frame))
                Semantic_masks.append(np.zeros_like(frame)[:,:,0])
        if mask_type == MaskType.Semantic_mask or mask_type == MaskType.Semantic_expansion_mask:
            return mask_container, Semantic_masks

    if mask_type == MaskType.bbox_mask:
        boxes_masks = []
        box_container = []

        info_index = 0

        for index, frame in enumerate(video):
            if index in frame_indexes:
                bbox = infos[info_index]['box']
                info_index = info_index + 1

                boxed_frame = get_bbox_image(frame, bbox, object_info['id'])
                box_container.append(boxed_frame)
                boxmask = np.zeros_like(frame)[:,:,0]
                boxmask[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])] = 1
                boxes_masks.append(boxmask)
            else:
                box_container.append(frame)
                boxes_masks.append(np.zeros_like(frame)[:,:,0])
        
        return box_container, boxes_masks

    if mask_type == MaskType.background_mask:
        bg_container = select_bg_from_video(background_masks, video)
        return bg_container, background_masks
    
    if mask_type == MaskType.fixed_mask or mask_type == MaskType.fixed_bg_mask:
        fixed_mask_container = []
        fixed_masks = []

        box = get_random_box(video[0])
        for index , frame in enumerate(video):
            if mask_type == MaskType.fixed_mask:
                boxed_frame = frame.copy()
                boxed_frame[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = 0
                fixed_mask_container.append(boxed_frame)

                fixed_mask = np.zeros_like(frame)[:,:,0]
                fixed_mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = 1
                fixed_masks.append(fixed_mask)
            if mask_type == MaskType.fixed_bg_mask:
                boxed_frame = frame.copy()

                fixed_mask = np.zeros_like(frame)[:,:,0]
                fixed_mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = 1
                fixed_mask = 1 - fixed_mask
                fixed_masks.append(fixed_mask)

                boxed_bg_frame = get_masked_image(boxed_frame, fixed_mask)
                fixed_mask_container.append(boxed_bg_frame)

        return fixed_mask_container, fixed_masks



def video_to_tensor(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将 BGR 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为张量并添加到帧列表中
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        frames.append(frame_tensor)
    
    cap.release()
    
    # 将所有帧组合成一个四维张量
    video_tensor = torch.stack(frames)  # (T, C, H, W)
    
    return video_tensor


def ndarray_to_pil(image: np.ndarray) -> Image:
    if np.max(image) <= 1.1:
        image = image * 255
    image = image.astype(np.uint8)
    return Image.fromarray(image)

def get_random_type():
    # 数字列表
    mask_type = [MaskType.Semantic_mask, MaskType.bbox_mask, MaskType.background_mask, MaskType.fixed_mask, MaskType.Semantic_expansion_mask, MaskType.fixed_bg_mask]

    # 概率权重列表（总和应为 1 或任意正数比例）
    weights = [0.3, 0.2, 0.1, 0.1, 0.2, 0.1]  # 例如，第一个数字1的概率是0.1

    # 从1-6中按设定概率随机选择一个数字
    chosen_number = random.choices(mask_type, weights=weights)[0]

    return chosen_number

def get_mask_tensor(video_tensor,mask_type,yolomodel):

    # return video_tensor,video_tensor

    masked_video_container,masks_container = get_mask(video_tensor,mask_type,yolomodel)
    
    masked_frames = [torch.from_numpy(frame.transpose(2,0,1)) for frame in masked_video_container]
    masked_video = torch.stack(masked_frames)

    masks = [torch.from_numpy(mask.reshape(1,mask.shape[0],mask.shape[1])) for mask in masks_container]
    # import ipdb; ipdb.set_trace()
    mask = torch.stack(masks)

    return masked_video,mask



class MaskProcessor:
    def __init__(self,args,YOLOmodel):
        # ratio
        # transform
        self.num_frames = args.num_frames
        if self.num_frames != 1:
            # inpaint
            self.t2v_ratio = args.t2v_ratio
            self.i2v_ratio = args.i2v_ratio
            self.transition_ratio = args.transition_ratio
            self.v2v_ratio = args.v2v_ratio
            self.clear_video_ratio = args.clear_video_ratio
            self.Semantic_ratio = args.Semantic_ratio
            self.bbox_ratio = args.bbox_ratio
            self.background_ratio = args.background_ratio
            self.fixed_ratio = args.fixed_ratio
            self.Semantic_expansion_ratio = args.Semantic_expansion_ratio
            self.fixed_bg_ratio = args.fixed_bg_ratio
            assert self.t2v_ratio + self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio + self.Semantic_ratio + self.bbox_ratio + self.background_ratio + self.fixed_ratio + self.fixed_bg_ratio + self.Semantic_expansion_ratio < 1, 'The sum of t2v_ratio, i2v_ratio, transition_ratio, v2v_ratio and clear video ratio should be less than 1.'
        
        self.min_clear_ratio = 0.0 if args.min_clear_ratio is None else args.min_clear_ratio
        assert self.min_clear_ratio >= 0 and self.min_clear_ratio <= 1, 'min_clear_ratio should be in the range of [0, 1].'


        self.transform = transforms.Compose([
            ToTensorVideo(),
            ae_norm[args.ae]
        ])

        self.init_mask_func()
        self.init_ratio()

        self.default_text_ratio = args.default_text_ratio

        self.yolomodel = YOLOmodel

    def init_mask_func(self):
        # mask: ones_like (t 1 h w)
        def t2iv(mask):
            mask[:] = 1
            return mask
        
        def i2v(mask):
            mask[0] = 0
            return mask
        
        def transition(mask):
            mask[0] = 0
            mask[-1] = 0
            return mask
        
        def v2v(mask):
            end_idx = random.randint(int(mask.shape[0] * self.min_clear_ratio), mask.shape[0])
            mask[:end_idx] = 0
            return mask
        
        def clear(mask):
            mask[:] = 0
            return mask
        
        def random_mask(mask):
            num_to_select = random.randint(int(mask.shape[0] * self.min_clear_ratio), mask.shape[0])
            selected_indices = random.sample(range(mask.shape[0]), num_to_select)
            mask[selected_indices] = 0
            return mask
        
        def Semantic_mask(video_tensor):
            return get_mask_tensor(video_tensor,MaskType.Semantic_mask,self.yolomodel)

        def bbox_mask(video_tensor):
            return get_mask_tensor(video_tensor,MaskType.bbox_mask,self.yolomodel)
        
        def background_mask(video_tensor):
            return get_mask_tensor(video_tensor,MaskType.background_mask,self.yolomodel)
        
        def fixed_mask(video_tensor):
            return get_mask_tensor(video_tensor,MaskType.fixed_mask,self.yolomodel)

        def Semantic_expansion_mask(video_tensor):
            return get_mask_tensor(video_tensor,MaskType.Semantic_expansion_mask,self.yolomodel)
        
        def fixed_bg_mask(video_tensor):
            return get_mask_tensor(video_tensor,MaskType.fixed_bg_mask,self.yolomodel)



        self.mask_functions = {
            MaskType.t2iv_mask: t2iv,
            MaskType.i2v_mask: i2v,
            MaskType.transition_mask: transition,
            MaskType.v2v_mask: v2v,
            MaskType.clear_mask: clear,
            MaskType.random_mask: random_mask,
            MaskType.Semantic_mask:Semantic_mask,
            MaskType.bbox_mask:bbox_mask,
            MaskType.background_mask:background_mask,
            MaskType.fixed_mask:fixed_mask,
            MaskType.Semantic_expansion_mask:Semantic_expansion_mask,
            MaskType.fixed_bg_mask:fixed_bg_mask
        }

    def init_ratio(self):

        self.mask_func_weights_video = {
            MaskType.t2iv_mask: self.t2v_ratio, 
            MaskType.i2v_mask: self.i2v_ratio, 
            MaskType.transition_mask: self.transition_ratio, 
            MaskType.v2v_mask: self.v2v_ratio, 
            MaskType.clear_mask: self.clear_video_ratio, 
            MaskType.Semantic_mask:self.Semantic_ratio,
            MaskType.bbox_mask:self.bbox_ratio,
            MaskType.background_mask:self.background_ratio,
            MaskType.fixed_mask:self.fixed_ratio,
            MaskType.Semantic_expansion_mask:self.Semantic_expansion_ratio,
            MaskType.fixed_bg_mask:self.fixed_bg_ratio,
            MaskType.random_mask: 1 - self.t2v_ratio - self.i2v_ratio - self.transition_ratio - self.v2v_ratio - self.clear_video_ratio - self.Semantic_ratio - self.bbox_ratio - self.background_ratio - self.fixed_ratio - self.Semantic_expansion_ratio - self.fixed_bg_ratio
            
        }

        self.mask_func_weights_image = {
            't2iv': 0.9, 
            'clear': 0.1
        }

    # t c h w
    def __call__(self,pixel_values):
        # pixel_values shape (T, C, H, W)
        # 1 means masked, 0 means not masked
        t, c, h, w = pixel_values.shape
        mask = torch.ones([t, 1, h, w], device=pixel_values.device, dtype=pixel_values.dtype)
        
        mask_func_name = random.choices(list(self.mask_func_weights_video.keys()), list(self.mask_func_weights_video.values()))[0]
        frame_mask_list = [MaskType.t2iv_mask,MaskType.i2v_mask,MaskType.transition_mask,MaskType.v2v_mask,MaskType.clear_mask,MaskType.random_mask]
        pos_mask_list = [MaskType.Semantic_mask,MaskType.bbox_mask,MaskType.background_mask,MaskType.fixed_mask,MaskType.Semantic_expansion_mask,MaskType.fixed_bg_mask]


        if mask_func_name in frame_mask_list:
            mask = self.mask_functions[mask_func_name](mask)
            masked_pixel_values = pixel_values * (mask < 0.5)
        
        if mask_func_name in pos_mask_list:
            masked_pixel_values,mask = self.mask_functions[mask_func_name](pixel_values)
        # save_video(masked_pixel_values.permute(0, 2, 3, 1).cpu().numpy(), 'masked_video.mp4')

        # import ipdb; ipdb.set_trace()

        pixel_values = self.transform(pixel_values)
        masked_pixel_values = self.transform(masked_pixel_values.to(torch.uint8))



        return masked_pixel_values, pixel_values, mask
    

