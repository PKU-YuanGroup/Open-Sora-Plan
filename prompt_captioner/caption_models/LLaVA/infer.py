from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import datetime, time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os.path as osp
import json, os
import torch
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from random import randint
import decord
from PIL import Image
import re
import math 

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates, SeparatorStyle

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

class my_dataset(Dataset):

    def __init__(self, args, image_processor, model):
        super().__init__()
        self.args = args
        self.shuffle = True
        self.image_processor = image_processor
        self.model = model
        # self.resolution = args.resolution
        assert args.train_file.endswith('.json'), 'train_file format error!'
        self.no_caption_id_list = []

        if not osp.exists(args.caption_save_dir):
            os.makedirs(args.caption_save_dir, exist_ok=True)

        with open(args.train_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            self.id_list = list(self.data.keys())
            for idx, id in enumerate(self.id_list):
                caption_json = osp.join(args.caption_save_dir, f'{id}_caption.json')
                if not os.path.exists(caption_json):
                    self.no_caption_id_list.append(id)
            print(f'Nums of no_caption_id_list is {len(self.no_caption_id_list)}, first id:{self.no_caption_id_list[0]}')

        print('Dataset nums is {}'.format(self.__len__()))

    def __len__(self):
        return len(self.no_caption_id_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, idx):
        if idx >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(idx + 1)

    def skip_sample(self, idx):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(idx=idx)

    def get_frames_from_video(self, video_path=None, frame_step=20):
        video_path = video_path
        vr = decord.VideoReader(video_path) 
        num_frames = len(vr) // frame_step
        # (h, w, c)   
        Image_frames = []
        Image_sizes = []
        for i in range(num_frames):
            idx = i * frame_step
            decord_frame = vr[idx].asnumpy()
            Image_frame = Image.fromarray(decord_frame) # Image.fromarray(array)  vs. numpy.asarray(Image) 
            Image_frames.append(Image_frame)
        Image_sizes.append(Image_frame.size)
        images_tensor = process_images(
                    Image_frames,
                    self.image_processor,
                    self.model.config
                ) 
        # [num_frames,3,336,336] [w,h]
        return images_tensor, Image_sizes

    def __getitem__(self, idx):
        try:
            video_id = self.no_caption_id_list[idx]
            video_path = self.data[video_id]
            caption_video_json =  osp.join(args.caption_save_dir, f'{id}_caption.json')
            if os.path.exists(caption_video_json):
                print('parallel task has process it :{}'.format(caption_video_json))
                return self.skip_sample(idx)
            if not osp.exists(video_path):
                print('video {} is not exists and skip this idx! '.format(video_path))
                return self.skip_sample(idx)
            images_tensor, Image_sizes = self.get_frames_from_video(video_path = video_path, frame_step = args.frame_step)
            return video_id, video_path, images_tensor, Image_sizes    
        except Exception as e:
            print('Read video error in {},{} and we have skip this !, this will not cause error!'.format(idx,e))
            # import ipdb; ipdb.set_trace()
            return self.skip_sample(idx)

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def collate_fn(batch):
    video_ids, video_paths,  images_tensor_tuple, Image_sizes_tuple = [], [], [], []
    for video_id, video_path, images_tensor, Image_sizes  in batch:
        video_ids.append(video_id)
        video_paths.append(video_path)
        images_tensor_tuple.append(images_tensor)
        Image_sizes_tuple.append(Image_sizes)
    return video_ids, video_paths,  images_tensor_tuple, Image_sizes_tuple
        

def llava_captioning(args):    
    # == model initialization ==
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=None,
            model_name=get_model_name_from_path(args.model_path),
            
            load_8bit = args.load_8bit, 
            load_4bit = args.load_4bit
        )
    # model = model.cuda(args.local_rank) # .set_device()

    train_dataset = my_dataset(args, image_processor, model)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank
    )
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             collate_fn = collate_fn,
                                             batch_size=args.vid_batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             sampler=train_sampler,
                                             drop_last=False,
                                             )
    #  images_tensor_tuple: bs*(frames,3,336,336)
    for index, (video_ids, video_paths,  images_tensor_tuple, Image_sizes_tuple) in enumerate(dataloader):
        
        for vid_idx, (video_id, video_path, images_tensor, Image_sizes) in enumerate(zip(video_ids, video_paths,  images_tensor_tuple, Image_sizes_tuple)):
            print(f'vid_idx:{vid_idx}, video_id:{video_id}!!')
            save_json = osp.join(args.caption_save_dir, f'{video_id}.json')
            if os.path.exists(save_json):
                continue
            # == code from run_llava.py ==
            qs = args.query
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                    
                    
            model_name = get_model_name_from_path(args.model_path)
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print(
                    "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                        conv_mode, args.conv_mode, args.conv_mode
                    )
                )
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images_tensor = images_tensor.to(model.device, dtype=torch.float16)
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .repeat(images_tensor.shape[0], 1) # repeat question for each frame
                .cuda(args.local_rank)
            ) # (bs, len)

            infer_times_per_video = math.ceil(images_tensor.shape[0]/args.batch_size)
            video_captions = []
            
            for infer_idx in range(infer_times_per_video):
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids[args.batch_size * infer_idx : args.batch_size * (infer_idx+1)],
                        images=images_tensor[args.batch_size * infer_idx : args.batch_size * (infer_idx+1)],
                        image_sizes=Image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                video_captions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
            
            with open(save_json, 'w', encoding='utf-8') as f:
                json.dump({video_id:video_captions}, f, indent=2)
            


def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=5400))
    torch.distributed.barrier()


def main(args):
    init_distributed_mode(args)
    t1=time.time()
    # start captioning
    llava_captioning(args)

    t2 = time.time()
    if args.rank == 0:
        print('Time : ',t2-t1,' s')
    dist.destroy_process_group()  
    


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--vid_batch_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--vat_root', type=str,default=None)
    parser.add_argument('--caption_save_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--frame_step', type=int, default=20)
    parser.add_argument('--model_path', type=str, required=True)
    # frame_step
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')  # --dist_on_itp   ddp
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, help='url used to set up distributed training')
    parser.add_argument('--gpus', default=0, help='DP CUDA devices')

    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)


    
