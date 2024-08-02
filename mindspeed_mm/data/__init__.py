from mindspeed_mm.data.dataloader.dataloader import (
    prepare_base_dataloader,
    prepare_sampler_dataloader,
    prepare_variable_dataloader,
)
from mindspeed_mm.data.datasets.image_dataset import ImageDataset
from mindspeed_mm.data.datasets.t2i_dataset import T2IDataset
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset
from mindspeed_mm.data.datasets.video_dataset import VideoDataset


def build_mm_dataset(args):
    """
    Build a multimodal dataset based on different tasks

    Args:
        dataset_param_dict
    Return:
        dataset
    """
    dataset_type = args.dataset_param_dict["dataset_type"]
    if dataset_type == "t2v":
        t2v_param = args.dataset_param_dict
        basic_param = t2v_param.get("basic_parameters", dict())
        vid_img_process = t2v_param.get("preprocess_parameters", dict())
        return T2VDataset(basic_param, vid_img_process, **t2v_param)
    elif dataset_type == "t2i":
        t2i_param = args.dataset_param_dict
        basic_param = t2i_param.get("basic_parameters", dict())
        vid_img_process = t2i_param.get("preprocess_parameters", dict())
        return T2IDataset(basic_param, vid_img_process, **t2i_param)
    elif dataset_type == "video":
        video_param = args.dataset_param_dict
        basic_param = video_param.get("basic_parameters", dict())
        vid_process = video_param.get("preprocess_parameters", dict())
        return VideoDataset(basic_param, vid_process, **video_param)
    elif dataset_type == "image":
        image_param = args.dataset_param_dict
        basic_param = image_param.get("basic_parameters", dict())
        img_process = image_param.get("preprocess_parameters", dict())
        return ImageDataset(basic_param, img_process, **image_param)
    else:
        raise NotImplementedError(dataset_type)


def build_mm_dataloader(dataset, args):
    """
    Build a multimodal dataloader based on different tasks

    dataloader_type interpretation:
    base: raw dataloader based on torch.utils.data.DataLoader
    sampler: prepare a dataloader for distributed training by building a specific sampler
    variable: used for variable dataset

    Args:
        dataloader_param_dict
    Return:
        dataloader
    """
    dataloader_param = args.dataloader_param_dict
    if dataloader_param["dataloader_mode"] == "base":
        data_loader = prepare_base_dataloader(dataset, **dataloader_param)
        return data_loader
    elif dataloader_param["dataloader_mode"] == "sampler":
        data_loader = prepare_sampler_dataloader(dataset, **dataloader_param)
        return data_loader
    elif dataloader_param["dataloader_mode"] == "variable":
        data_loader = prepare_variable_dataloader(dataset, **dataloader_param)
        return data_loader
    else:
        raise NotImplementedError(dataloader_param["dataloader_mode"])
