""" ParamToParamDataset class """

from torch.utils.data.dataset import T_co

from ..dataset_handler import DatasetHandler
from .prediction_dataset import PredictionDataset


class ParamToParamDataset(PredictionDataset):
    """ ParamToParamDataset

        Implements the PredictionDataset abstract base class (ABC).
        Provides the pytorch dataset for parameter to parameter prediction.
    """

    def __init__(self, dataset_handler: DatasetHandler):
        super().__init__(dataset_handler)
        self.train_labels = dataset_handler.contiguous_experiments
        self.input_dim = 46  # TODO: remove magic number
        self.output_dim = 46  # TODO: remove magic number

    def __getitem__(self, index) -> T_co:
        """
        provides feature and target vector of the dataset
        :param index: dataset index
        :return: tensor of data
        """
        feature_vec = self._dataset_handler.get_experiment_params_by_id(self.train_labels[index][0])
        target_vec = self._dataset_handler.get_experiment_params_by_id(self.train_labels[index][1])
        return self.transform_input_output_vectors_to_tensor(feature_vec, target_vec)
