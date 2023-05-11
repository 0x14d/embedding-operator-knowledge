""" DatasetHandler class """

# pylint: disable=import-error

import os
import sys
import ast
from typing import Optional, Tuple

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler

from data_provider.abstract_data_provider import AbstractDataProvider
from data_provider.data_provider_singleton import get_data_provider
from knowledge_infusion.dataset.utils.dataset_preparation import prepare_dataframe

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class DatasetHandler:
    """
    Provides the experiment data from DataProvider and offers different data operations
    """

    dataset_directory: str

    def __init__(
        self,
        dataset_type: str,
        data_provider: str,
        dataset_path: Optional[str] = None,
        **kwargs
    ):
        if data_provider is not None:
            self._data_provider = get_data_provider(
                data_provider,
                ignore_singleton=data_provider == 'synthetic',
                **kwargs
            )
        else:
            self._data_provider = None

        self._load_dataset(dataset_path)

        self._rating_normalizer = MinMaxScaler()
        self._param_normalizer = MinMaxScaler()
        self._experiment_ratings = DataFrame()
        self._experiment_parameters = DataFrame()
        self._dataset_type = dataset_type

        if not self._dataframe.empty:
            self._experiment_parameters, self._experiment_ratings = prepare_dataframe(
                dataframe=self._dataframe,
                info=self._info,
                boolean_parameters=self._boolean_parameters,
                contiguous_experiments=self.contiguous_experiments,
                dataset_type=self._dataset_type,
                param_normalizer=self.param_normalizer,
                rating_normalizer=self.rating_normalizer,
                fit_normalizer=True
            )
        else:
            print('ERROR: Database is not reachable!')

    @property
    def contiguous_experiments(self):
        return self._contiguous_experiments

    @property
    def rating_normalizer(self):
        return self._rating_normalizer

    @property
    def param_normalizer(self):
        return self._param_normalizer

    @property
    def experiment_ratings(self):
        return self._experiment_ratings

    @property
    def parameters(self):
        return list(self._experiment_parameters.columns)[:-1]

    @property
    def ratings(self):
        return list(self._experiment_ratings.columns)[:-1]

    @property
    def data_provider(self) -> AbstractDataProvider:
        return self._data_provider

    def get_idx_from_rating(self, rating: str):
        pass

    def _load_dataset(self, dataset_path: str) -> None:
        """
        Loads the dataset from the given path if it exists.
        Othwerwise it loads the dataset from the data provider.
        :param dataset_path: path to the dataset
        :return: None
        """
        if dataset_path is None:
            dataset_path = 'knowledge_infusion/dataset/'

        self.dataset_directory = dataset_path
        dataset_path += 'database/'

        if not os.path.isdir(dataset_path):
            os.makedirs(dataset_path)

        try:
            self._dataframe = pd.read_pickle(dataset_path + 'dataframe.pkl')
            with open(dataset_path + "info.txt", 'r', encoding='utf-8') as file:
                self._info = ast.literal_eval(file.read())
            with open(dataset_path + "boolean_parameters.txt", 'r', encoding='utf-8') as file:
                self._boolean_parameters = ast.literal_eval(file.read())
        except FileNotFoundError:
            print("INFO: Load Data from Dataprovider")
            # load dataframes with DataProvider
            self._dataframe, self._info, _, self._boolean_parameters, _, _ = \
                self._data_provider.get_executed_experiments_data(
                    completed_only=False,
                    labelable_only=False,
                    containing_insights_only=True,
                    include_oneoffs=True,
                    limit=436
                )
            self._dataframe.to_pickle(dataset_path + 'dataframe.pkl')
            self._dataframe.to_csv(dataset_path + 'dataframe.csv')
            with open(dataset_path + "info.txt", "w", encoding='utf-8') as out:
                out.write(str(self._info))
            with open(dataset_path + "boolean_parameters.txt", "w", encoding='utf-8') as out:
                out.write(str(self._boolean_parameters))
        try:
            with open(dataset_path + "contiguous_experiments.txt", 'r', encoding='utf-8') as file:
                self._contiguous_experiments = ast.literal_eval(file.read())
        except FileNotFoundError:
            print("INFO: Generate Contiguous Experiments from DataProvider")
            self._contiguous_experiments = self._data_provider.get_connected_experiments_by_last_influences(
                completed_only=False,
                labelable_only=False,
                containing_insights_only=True,
                include_oneoffs=True,
                limit=436
            )
            with open(dataset_path + "contiguous_experiments.txt", "w", encoding='utf-8') as out:
                out.write(str(self._contiguous_experiments))

    def get_experiment_params_by_id(self, experiment_id: int) -> DataFrame:
        """
        selects experiment parameters by experiment_id
        :param experiment_id: experiment identifier
        :return: experiment parameters
        """
        experiment = self._experiment_parameters.loc[self._dataframe['ID']
                                                     == experiment_id]
        experiment_without_id = experiment.drop(['ID'], axis=1)
        return experiment_without_id

    def get_experiment_ratings_by_id(self, experiment_id: int) -> DataFrame:
        """
        selects experiment ratings by experiment_id
        :param experiment_id: experiment identifier
        :return: experiment ratings
        """
        experiment = self._experiment_ratings.loc[self._dataframe['ID']
                                                  == experiment_id]
        experiment_ratings_without_id = experiment.drop(['ID'], axis=1)
        return experiment_ratings_without_id
