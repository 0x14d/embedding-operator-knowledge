""" PredictionDataset class """

import numpy as np
from abc import ABC
from typing import List, Tuple

from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from ..dataset_handler import DatasetHandler


class PredictionDataset(Dataset, ABC):
    """
    PredictionDataset
    Abstract Base Class (ABC) of Pytorch Dataset Class
    """
    _input_dim: int
    _output_dim: int

    def __init__(self, dataset_handler: DatasetHandler):
        self._dataset_handler = dataset_handler
        self._contiguous_experiments: List[Tuple[int, int]] = dataset_handler.contiguous_experiments
        self.transform = ToTensor()
        self.target_transform = ToTensor()

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @input_dim.setter
    def input_dim(self, dim):
        self._input_dim = dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @output_dim.setter
    def output_dim(self, dim):
        self._output_dim = dim

    def __len__(self):
        return len(self._contiguous_experiments)

    def transform_input_output_vectors_to_tensor(self, feature: DataFrame, target: DataFrame) -> (Tensor, Tensor):
        """
        Transform feature and target vector into Tensor with right dimensions
        :param feature: feature vector
        :param target: target vector
        :return: Tuple of feature tensor and target tensor
        """
        if self.transform:
            feature = self.transform(np.array(feature).astype(float))
        if self.target_transform:
            target = self.target_transform(np.array(target).astype(float))
        return feature.view(1, 1, self._input_dim), target.view(1, 1, self._output_dim)
