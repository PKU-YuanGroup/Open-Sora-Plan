
import os
from typing import Union

from mindspeed_mm.data.data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    TEXT,
    VIDEO,
    IMG_FPS
)
from mindspeed_mm.data.data_utils.utils import InpaintVideoProcesser, InpaintTextProcesser
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset, T2VOutputData



class InpaintDataset(T2VDataset):
    def __init__(
        self,
        basic_param: dict,
        vid_img_process: dict,
        use_text_processer: bool = False,
        use_clean_caption: bool = True,
        support_chinese: bool = False,
        model_max_length: int = 120,
        tokenizer_config: Union[dict, None] = None,
        use_feature_data: bool = False,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        mask_type_ratio_dict_video: dict = None,
        **kwargs,
    ):
   
        super().__init__(
            basic_param=basic_param,
            vid_img_process=vid_img_process,
            use_text_processer=use_text_processer,
            use_clean_caption=use_clean_caption,
            support_chinese=support_chinese,
            model_max_length=model_max_length,
            tokenizer_config=tokenizer_config,
            use_feature_data=use_feature_data,
            vid_img_fusion_by_splicing=vid_img_fusion_by_splicing,
            use_img_num=use_img_num,
            use_img_from_vid=use_img_from_vid,
            **kwargs,
        )

        assert self.num_frames > 0, "num_frames should be greater than 0 in inpaint mode."

        self.train_resize_pipeline = vid_img_process.get("train_resize_pipeline", None)
        self.min_clear_ratio = vid_img_process.get("min_clear_ratio", 0.0)
        self.max_clear_ratio = vid_img_process.get("max_clear_ratio", 1.0)
        self.default_text_ratio = vid_img_process.get("default_text_ratio", 0.5)

        # Covering the original value
        self.video_processer = InpaintVideoProcesser(
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            train_pipeline=self.train_pipeline,
            train_resize_pipeline=self.train_resize_pipeline,
            data_storage_mode=self.data_storage_mode,
            train_fps=self.train_fps,
            speed_factor=self.speed_factor,
            drop_short_ratio=self.drop_short_ratio,
            max_height=self.max_height,
            max_width=self.max_width,
            min_clear_ratio=self.min_clear_ratio,
            max_clear_ratio=self.max_clear_ratio,
            mask_type_ratio_dict_video=mask_type_ratio_dict_video,
        )

        if self.use_text_processer and tokenizer_config is not None:
            self.text_processer = InpaintTextProcesser(
                model_max_length=model_max_length,
                tokenizer=self.tokenizer,
                use_clean_caption=use_clean_caption,
                support_chinese=support_chinese,
                cfg=self.cfg,
                default_text_ratio=self.default_text_ratio
            )

