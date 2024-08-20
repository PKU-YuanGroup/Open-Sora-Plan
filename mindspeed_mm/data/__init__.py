from mindspeed_mm.data.dataloader.dataloader import (
    prepare_base_dataloader,
    prepare_sampler_dataloader,
    prepare_variable_dataloader,
)
from mindspeed_mm.data.datasets.image_dataset import ImageDataset
from mindspeed_mm.data.datasets.t2i_dataset import T2IDataset
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset
from mindspeed_mm.data.datasets.video_dataset import VideoDataset


def build_mm_dataset(dataset_param):
    """
    Build a multimodal dataset based on different tasks

    Args:
        dataset_param
    Return:
        dataset
    """

    dataset_param = dataset_param.to_dict()
    for check_key in ["dataset_type", "basic_parameters", "preprocess_parameters"]:
        if check_key not in dataset_param:
            raise AssertionError(f"Key parameter missing: {check_key}")
    dataset_type = dataset_param["dataset_type"]
    basic_param = dataset_param["basic_parameters"]
    preprocess_param = dataset_param["preprocess_parameters"]
    if dataset_type == "t2v":
        return T2VDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "t2i":
        return T2IDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "video":
        return VideoDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "image":
        return ImageDataset(basic_param, preprocess_param, **dataset_param)
    else:
        raise NotImplementedError(dataset_type)


def build_mm_dataloader(dataset, dataloader_param, process_group=None):
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

    dataloader_param = dataloader_param.to_dict()
    if "dataloader_mode" not in dataloader_param:
        raise AssertionError("Key parameter missing: dataloader_mode")
    dataloader_mode = dataloader_param["dataloader_mode"]
    if dataloader_mode == "base":
        data_loader = prepare_base_dataloader(dataset, **dataloader_param)
        return data_loader
    elif dataloader_mode == "sampler":
        data_loader = prepare_sampler_dataloader(
            dataset, **dataloader_param, process_group=process_group
        )
        return data_loader
    elif dataloader_mode == "variable":
        data_loader = prepare_variable_dataloader(dataset, **dataloader_param)
        return data_loader
    else:
        raise NotImplementedError(dataloader_param["dataloader_mode"])
