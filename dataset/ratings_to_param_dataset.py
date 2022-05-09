""" RatingsToParamDataset class"""
from pandas import DataFrame
from torch.utils.data.dataset import T_co

from dataset_handler import DatasetHandler
from dataset.prediction_dataset import PredictionDataset


class RatingsToParamDataset(PredictionDataset):
    """
    RatingsToParamDataset
    Implements the PredictionDataset abstract base class (ABC).
    Provides the pytorch dataset for ratings to parameter prediction.
    """

    def __init__(self, dataset_handler: DatasetHandler):
        super().__init__(dataset_handler)
        self._input_dim = 13
        self._output_dim = 46  # TODO: remove magic number

    def __getitem__(self, index) -> T_co:
        feature: DataFrame = self._dataset_handler.get_experiment_ratings_by_id(self._contiguous_experiments[index][0])
        target: DataFrame = self._dataset_handler.get_experiment_params_by_id(self._contiguous_experiments[index][1])
        return self.transform_input_output_vectors_to_tensor(feature, target)
