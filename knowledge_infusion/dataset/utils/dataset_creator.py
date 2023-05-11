"""This module provides functionality to create a `PredictionDataset`."""

# pylint: disable=import-error

from knowledge_infusion.dataset.dataset_handler import DatasetHandler
from knowledge_infusion.dataset.param_to_param_dataset import ParamToParamDataset
from knowledge_infusion.dataset.prediction_dataset import PredictionDataset
from knowledge_infusion.dataset.ratings_and_param_to_param_dataset \
    import RatingsAndParamToParamDataset
from knowledge_infusion.dataset.ratings_to_param_dataset import RatingsToParamDataset
from knowledge_infusion.utils.schemas import DatasetType


def create_dataset(dataset_handler: DatasetHandler, dataset_type: DatasetType) -> PredictionDataset:
    """Creates a `PredictionDataset` of the specified type using the specified dataset handler."""
    if dataset_type == 'param2param':
        return ParamToParamDataset(dataset_handler)
    if dataset_type == 'ratings2param':
        return RatingsToParamDataset(dataset_handler)
    if dataset_type == 'ratings&param2param':
        return RatingsAndParamToParamDataset(dataset_handler)
    raise ValueError(f'invalid dataset type {dataset_type}')
