from abc import abstractmethod
from typing import List
import torch

from ..model import ScoreModel

class VQAScoreModel(ScoreModel):

    @abstractmethod
    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str,
                answer_template: str) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        question_template: a string with optional {} to be replaced with the 'text'
        answer_template: a string with optional {} to be replaced with the 'text'
        """
        pass