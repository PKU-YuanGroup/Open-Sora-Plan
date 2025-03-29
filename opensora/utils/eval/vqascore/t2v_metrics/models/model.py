from abc import ABC, abstractmethod
from typing import List
import os
import torch
import numpy as np
from PIL import Image
    
from ..constants import HF_CACHE_DIR

def image_loader(image_path):
    if image_path.split('.')[-1] == 'npy':
        return Image.fromarray(np.load(image_path)[:, :, [2, 1, 0]], 'RGB')
    else:
        return Image.open(image_path).convert("RGB")

class ScoreModel(ABC):
    def __init__(self,
                 model_name='clip-flant5-xxl',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.image_loader = image_loader
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model, tokenizer, and etc.
        """
        pass

    @abstractmethod
    def load_images(self,
                    image: List[str]) -> torch.Tensor: 
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        pass

    @abstractmethod
    def forward(self,
                images: List[str],
                texts: List[str]) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        pass