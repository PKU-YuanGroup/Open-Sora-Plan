import os
import copy
from typing import Union
import torch

from mindspeed_mm.data.data_utils.utils import (
    ImageProcesser,
    preprocess
)
from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.models import Tokenizer


class ImageDataset(MMBaseDataset):
    """
    A multimodal dataset for supervised fine-tuning based on MMBaseDataset.

    Args:
        basic_param (dict): Basic parameters such as data_path, data_folder, etc.
        img_process (dict): some data preprocessing parameters.
        constants (dict): some data preprocessing constants.
        use_text_processer (bool): whether text preprocessing
        tokenizer_config (dict): The config of tokenizer.
        is_multimodal (bool): Flag to indicate if the model is multimodal (handles both text and images).
        mm_use_im_start_end (bool): Flag to indicate if the image start and end tokens should be used.
        template_name (str): The name of the template to be used.
        image_size (int): The size to which images will be resized.
        down_sample_ratio (float): The ratio by which to downsample the images.
        patch_size (int): The size of the patches to be used for processing images.
        group_by_length (bool): Flag to indicate if data should be grouped by length.
        dynamic_image_size (bool): Flag to indicate if the image size should be dynamically adjusted.
        use_thumbnail (bool): Flag to indicate if thumbnails should be used for images.
        min_dynamic_patch (int): The minimum number of dynamic patches.
        max_dynamic_patch (int): The maximum number of dynamic patches.
        repeat_time (int): The number of times to repeat the data processing.
    """

    def __init__(
            self,
            basic_param: dict,
            img_process: dict,
            use_text_processer: bool = False,
            tokenizer_config: Union[dict, None] = None,
            is_multimodal: bool = True,
            mm_use_im_start_end: bool = True,
            template_name: str = "",
            image_size: int = 224,
            down_sample_ratio: float = 0.5,
            patch_size: int = 14,
            group_by_length: bool = False,
            dynamic_image_size: bool = False,
            use_thumbnail: bool = False,
            min_dynamic_patch: int = 1,
            max_dynamic_patch: int = 6,
            repeat_time: int = 1,
            **kwargs,
    ):
        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.template_name = template_name
        self.image_size = image_size
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.patch_size = patch_size
        self.down_sample_ratio = down_sample_ratio
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (self.down_sample_ratio ** 2))

        if repeat_time < 1:
            # If repeat_time is less than 1, select a portion of the data
            self.data_samples = self.data_samples[:int(len(self.data_samples) * repeat_time)]
        if repeat_time > 1:
            # Repeat the list if repeat_time is greater than 1
            self.data_samples = self.data_samples * repeat_time

        self.is_multimodal = is_multimodal
        self.mm_use_im_start_end = mm_use_im_start_end
        self.train_pipeline = img_process.get("train_pipeline", None)
        self.image_reader_type = img_process.get("image_reader_type", "torchvision")
        self.image_processer_type = img_process.get("image_processer_type", "image2pixel")
        self.image_processer = ImageProcesser(train_pipeline=self.train_pipeline,
                                              image_processer_type=self.image_processer_type,
                                              image_reader_type=self.image_reader_type,
                                              dynamic_image_size=self.dynamic_image_size,
                                              image_size=self.image_size,
                                              min_dynamic_patch=self.min_dynamic_patch,
                                              max_dynamic_patch=self.max_dynamic_patch,
                                              use_thumbnail=self.use_thumbnail)
        if tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data_samples)

    def multi_modal_get_item(self, data_item):
        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = "<image>\n" + data_item["conversations"][0]["value"]

        image_path = os.path.join(self.data_folder, data_item["image"])
        pixel_values = self.image_processer(image_path, train_pipeline=self.train_pipeline,
                                            mode='single_image', num_image=1)
        num_patches = pixel_values.size(0)

        ret = preprocess(
            template_name=self.template_name,
            sources=copy.deepcopy([data_item["conversations"]]),
            tokenizer=self.tokenizer,
            num_image_token_list=[self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            is_multimodal=self.is_multimodal,
            mm_use_im_start_end=self.mm_use_im_start_end
        )

        ret["pixel_values"] = pixel_values
        ret["image_flags"] = torch.tensor([1] * num_patches, dtype=torch.long)

        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        pass

    def pure_text_get_item(self, data_item):
        pass

    def getitem(self, index):
        index = index % len(self.data_samples)
        data_item = self.data_samples[index]
        if "image" in data_item and len(data_item["image"]) != 0:
            if isinstance(data_item["image"], list):
                ret = None
            else:
                ret = self.multi_modal_get_item(data_item)
        else:
            ret = None
        return ret
