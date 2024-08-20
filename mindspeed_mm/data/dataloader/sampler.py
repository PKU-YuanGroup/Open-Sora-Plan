import random
import warnings
from collections import OrderedDict, defaultdict
from pprint import pprint
from typing import Iterator, List, Optional

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ..datasets.t2v_dataset import VariableT2VDataset


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


# TODO
class VariableVideoBatchSampler(DistributedSampler):
    pass
