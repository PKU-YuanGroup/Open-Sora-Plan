import os

from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, ToTensor, Compose, Resize, Normalize
from PIL import Image
import pickle as pkl
import json


def load_pkl_or_json(path: str):
    if path.endswith("pkl"):
        read_mode = "rb"
        load_func = pkl.load
    elif path.endswith("json"):
        read_mode = "r"
        load_func = json.load
    else:
        raise NotImplementedError

    with open(path, read_mode) as file:
        data = load_func(file)
    return data


class InversionValidImageDataset(Dataset):
    """
    For image valid.
    """

    def __init__(self, data_txt, resolution) -> None:
        self.dataset = []
        self.resolution = resolution

        self.transform = Compose(
            [
                ToTensor(),
                Resize(resolution),
                CenterCrop(resolution),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # output [-1, 1]
            ]
        )
    
        self._load_dataset(data_txt)

    def _load_dataset(self, data_txt):
        """Parse data_txt and load data index"""

        # Load file
        with open(data_txt, "r") as file:
            subsets = file.readlines()

        # Get subset
        for subset in subsets:
            data_base_dir, data_file = [text.strip() for text in subset.split(",")]
            subset_data = load_pkl_or_json(data_file)
            self.dataset += [
                (
                    os.path.join(data_base_dir, line["path"]),
                    line["cap"][0],
                )
                for line in subset_data
            ]

    def __getitem__(self, index):
        image_path, caption = self.dataset[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "caption": caption
        }

    def __len__(self):
        return len(self.dataset)
