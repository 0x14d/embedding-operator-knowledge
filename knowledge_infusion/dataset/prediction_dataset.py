""" PredictionDataset class """

# pylint: disable=import-error

from abc import ABC
from typing import List, Optional, Tuple

import igraph
import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorConfig
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from knowledge_extraction.rule_to_representation import get_graph_from_data_provider
from knowledge_infusion.dataset.dataset_handler import DatasetHandler


class PredictionDataset(Dataset, ABC):
    """
    PredictionDataset
    Abstract Base Class (ABC) of Pytorch Dataset Class
    """
    _input_dim: int
    _output_dim: int
    _rating_dim: int
    _parameter_dim: int

    def __init__(self, dataset_handler: DatasetHandler):
        self._dataset_handler = dataset_handler
        self._contiguous_experiments: List[Tuple[int, int]] = dataset_handler.contiguous_experiments
        self.transform = ToTensor()
        self.target_transform = ToTensor()

        self.parameter_dim = 46#
        self.rating_dim = 13

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

    @property
    def rating_dim(self) -> int:
        return self._rating_dim

    @rating_dim.setter
    def rating_dim(self, dim):
        self._rating_dim = dim

    @property
    def parameter_dim(self) -> int:
        return self._output_dim

    @parameter_dim.setter
    def parameter_dim(self, dim):
        self._parameter_dim = dim

    @property
    def dataset_handler(self) -> DatasetHandler:
        return self._dataset_handler

    def __len__(self):
        return len(self._contiguous_experiments)

    def transform_input_output_vectors_to_tensor(
        self,
        feature: DataFrame,
        target: DataFrame
    ) -> Tuple[Tensor, Tensor]:
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

    def get_knowledge_graph(
        self,
        sdg_config: Optional[SdgConfig] = None,
        kg_config: Optional[KnowledgeGraphGeneratorConfig] = None
    ) -> igraph.Graph:
        """
        Returns the knowledge graph of the dataset.
        :param sdg_config: Optional sdg config to use for the kg generation
        :param kg_config: Optional kg config to use for the kg generation
        :return: Generated kg
        """
        return get_graph_from_data_provider(
            self._dataset_handler.data_provider,
            sdg_config,
            kg_config
        )
