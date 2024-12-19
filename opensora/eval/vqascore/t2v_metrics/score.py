from abc import abstractmethod
from typing import List, TypedDict, Union
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from .constants import HF_CACHE_DIR

class ImageTextDict(TypedDict):
    images: List[str]
    texts: List[str]

class Score(nn.Module):

    def __init__(self,
                 model: str,
                 device: str='cuda',
                 cache_dir: str=HF_CACHE_DIR,
                 **kwargs):
        """Initialize the ScoreModel
        """
        super().__init__()
        # assert model in self.list_all_models()
        self.device = device
        self.model = self.prepare_scoremodel(model, device, cache_dir, **kwargs)
    
    @abstractmethod
    def prepare_scoremodel(self,
                           model: str,
                           device: str,
                           cache_dir: str,
                           **kwargs):
        """Prepare the ScoreModel
        """
        pass
    
    @abstractmethod
    def list_all_models(self) -> List[str]:
        """List all available models
        """
        pass

    def forward(self,
                images: Union[str, List[str]],
                texts: Union[str, List[str]],
                **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        if type(images) == str:
            images = [images]
        if type(texts) == str:
            texts = [texts]
        
        scores = torch.zeros(len(images), len(texts)).to(self.device)
        for i, image in enumerate(images):
            scores[i] = self.model.forward([image] * len(texts), texts, **kwargs)
        return scores
    
    def batch_forward(self,
                      dataset: List[ImageTextDict],
                      batch_size: int=16,
                      **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        num_samples = len(dataset)
        num_images = len(dataset[0]['images'])
        num_texts = len(dataset[0]['texts'])
        scores = torch.zeros(num_samples, num_images, num_texts).to(self.device)
        
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        counter = 0
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            cur_batch_size = len(batch['images'][0])
            assert len(batch['images']) == num_images, \
                f"Number of image options in batch {batch_idx} is {len(batch['images'])}. Expected {num_images} images."
            assert len(batch['texts']) == num_texts, \
                f"Number of text options in batch {batch_idx} is {len(batch['texts'])}. Expected {num_texts} texts."
            
            for image_idx in range(num_images):
                images = batch['images'][image_idx]
                for text_idx in range(num_texts):
                    texts = batch['texts'][text_idx]
                    scores[counter:counter+cur_batch_size, image_idx, text_idx] = \
                        self.model.forward(images, texts, **kwargs)
            
            counter += cur_batch_size
        return scores
    