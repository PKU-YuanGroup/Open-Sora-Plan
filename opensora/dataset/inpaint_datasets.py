
import os, random
import numpy as np

import torch
from accelerate.logging import get_logger

from opensora.utils.utils import text_preprocessing

from opensora.dataset.t2v_datasets import SingletonMeta, DataSetProg
from opensora.dataset.t2v_datasets import T2V_dataset

logger = get_logger(__name__)

dataset_prog = DataSetProg()


class Meta_dataset(T2V_dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop):
        super().__init__(args, transform, temporal_sample, tokenizer, transform_topcrop)

        if self.num_frames != 1:
            # inpaint
            # The proportion of executing the i2v task.
            self.i2v_ratio = args.i2v_ratio
            self.transition_ratio = args.transition_ratio
            self.v2v_ratio = args.v2v_ratio
            self.clear_video_ratio = args.clear_video_ratio
            assert self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio < 1, 'The sum of i2v_ratio, transition_ratio, v2v_ratio and clear video ratio should be less than 1.'
        
        self.default_text_ratio = args.default_text_ratio
        self.default_text = f"The {'video' if self.num_frames != 1 else 'image'} showcases a scene with coherent and clear visuals."
    
    def get_mask_masked_video(self, video):
        # video shape (T, C, H, W)
        # 1 means masked, 0 means not masked
        t, c, h, w = video.shape
        mask = torch.ones_like(video, device=video.device, dtype=video.dtype)
        
        rand_num = random.random()
        # i2v
        if rand_num < self.i2v_ratio:
            mask[0] = 0
        # transition
        elif rand_num < self.i2v_ratio + self.transition_ratio:
            mask[0] = 0
            mask[-1] = 0
        # video continuation
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.v2v_ratio:
            end_idx = random.randint(1, t)
            mask[:end_idx] = 0
        # clear video
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio:
            mask[:] = 0
        # random mask
        else:
            idx_to_select = random.randint(0, t - 1)
            selected_indices = random.sample(range(0, t), idx_to_select)
            mask[selected_indices] = 0
        masked_video = video * (mask < 0.5)

        # save_video(masked_video.permute(0, 2, 3, 1).cpu().numpy(), 'masked_video.mp4')
        return dict(mask=mask, masked_video=masked_video)

    def drop(self, text):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''

        return dict(text=text)

    
class Inpaint_dataset(Meta_dataset):
    def get_video(self, idx):
        video_path = dataset_prog.cap_list[idx]['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indice = dataset_prog.cap_list[idx]['sample_frame_index']
        video = self.decord_read(video_path, predefine_num_frames=len(frame_indice))

        h, w = video.shape[-2:]
        assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]
        video = self.transform(video)  # T C H W -> T C H W

        # inpaint
        inpaint_cond_data = self.get_mask_masked_video(video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_video']
        video = torch.cat([video, masked_video, mask], dim=1) # T 3*C H W

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = dataset_prog.cap_list[idx]['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text_preprocessing(text, support_Chinese=self.support_Chinese)

        text = self.drop(text)['text']

        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']
        return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask)
    