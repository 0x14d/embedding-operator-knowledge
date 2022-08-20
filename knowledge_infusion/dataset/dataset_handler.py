""" DatasetHandler class """

import os
import sys
import ast
from typing import Optional, Tuple

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class DatasetHandler:
    """
    Provides the experiment data from DataProvider and offers different data operations
    """

    def __init__(self, dataset_type: str, data_provider: str, dataset_path: Optional[str] = None, **kwargs):
        # load dataframes with DataProvider
        self._data_provider = get_data_provider(
            data_provider,
            ignore_singleton=data_provider == 'synthetic',
            **kwargs
        )

        if dataset_path is None:
            dataset_path = 'knowledge_infusion/dataset/database/'
        else:
            dataset_path += 'database/'

        if not os.path.isdir(dataset_path):
            os.makedirs(dataset_path)

        try:
            self._dataframe = pd.read_pickle(dataset_path + 'dataframe.pkl')
            with open(dataset_path + "info.txt", 'r') as f:
                self._info = ast.literal_eval(f.read())
            with open(dataset_path + "boolean_parameters.txt", 'r') as f:
                self._boolean_parameters = ast.literal_eval(f.read())
        except FileNotFoundError:
            print("INFO: Load Data from MongoDB")
            self._dataframe, self._info, _, self._boolean_parameters, _, _ = \
                self._data_provider.get_executed_experiments_data(
                    completed_only=False, labelable_only=False, containing_insights_only=True, include_oneoffs=True,
                    limit=436)
            self._dataframe.to_pickle(dataset_path + 'dataframe.pkl')
            with open(dataset_path + "info.txt", "w") as outfile:
                outfile.write(str(self._info))
            with open(dataset_path + "boolean_parameters.txt", "w") as outfile:
                outfile.write(str(self._boolean_parameters))
        try:
            with open(dataset_path + "contiguous_experiments.txt", 'r') as f:
                self._contiguous_experiments = ast.literal_eval(f.read())
        except FileNotFoundError:
            print("INFO: Generate Contiguous Experiments from DataProvider")
            self._contiguous_experiments = self._data_provider.get_connected_experiments_by_last_influences(
                completed_only=False, labelable_only=False, containing_insights_only=True, include_oneoffs=True,
                limit=436)
            with open(dataset_path + "contiguous_experiments.txt", "w") as outfile:
                outfile.write(str(self._contiguous_experiments))

        self._rating_normalizer = MinMaxScaler()
        self._param_normalizer = MinMaxScaler()
        self._experiment_ratings = DataFrame()
        self._experiment_parameters = DataFrame()
        self._dataset_type = dataset_type

        if not self._dataframe.empty:
            self._filter_data_by_info_tag('process_parameters')
            self._filter_data_by_info_tag('ratings')
            self._check_experiments_completeness()
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

    def get_idx_from_rating(self, rating: str):
        pass

    def _filter_data_by_info_tag(self, info_tag: str) -> None:
        """
        filters the data either by experiment process_parameters or by ratings
        :param info_tag: process_parameters or ratings
        :return: None
        """
        if info_tag == 'process_parameters':
            process_parameters = [process_param for process_param in list(self._info[info_tag])
                                  if process_param in self._dataframe.columns and
                                  process_param not in self._boolean_parameters]
            process_params_df = self._dataframe.loc[:, process_parameters]
            # delete columns with string
            for column in process_params_df:
                element = process_params_df.loc[0, column]
                if isinstance(element, str):
                    process_params_df.drop([column], axis=1, inplace=True)

            self._experiment_parameters = self._normalize_dataframe_and_add_id(process_params_df, info_tag)
        elif info_tag == 'ratings':
            ratings = list(self._info[info_tag])
            ratings.remove('overall_ok')
            ratings_df = self._dataframe.loc[:, ratings]
            self._experiment_ratings = self._normalize_dataframe_and_add_id(ratings_df, info_tag)
        else:
            print(f'ERROR: Info tag: {info_tag} is invalid!')

    def _normalize_dataframe_and_add_id(self, dataframe: DataFrame, info_tag: str) -> DataFrame:
        """
        normalize dataframe and add ID column
        :param dataframe: input dataframe
        :return: output dataframe
        """
        if info_tag == 'process_parameters':
            process_params_df = pd.DataFrame(self._param_normalizer.fit_transform(dataframe),
                                             columns=dataframe.columns)
        else:
            process_params_df = pd.DataFrame(self._rating_normalizer.fit_transform(dataframe),
                                             columns=dataframe.columns)
        process_params_df['ID'] = self._dataframe.loc[:, self._dataframe.columns == 'ID']
        return process_params_df.copy()

    def _check_experiments_completeness(self) -> None:
        """
        check if contiguous experiments hold all necessary data in the dataframe and provide correct tensor shape
        :return: None
        """
        feature_experiment: DataFrame or None = None
        expected_input_shape: Tuple or None = None
        expected_output_shape: Tuple = (1, 47)  # TODO: Remove magic numbers
        for experiment_tuple in self._contiguous_experiments:
            if self._dataset_type == 'param2param':
                expected_input_shape = (1, 47)  # TODO: Remove magic numbers
                feature_experiment = self._experiment_parameters.loc[
                    self._experiment_parameters['ID'] == experiment_tuple[0]]
            elif self._dataset_type == 'ratings2param':
                expected_input_shape = (1, 14)  # TODO: Remove magic numbers
                feature_experiment = self._experiment_ratings.loc[
                    self._experiment_ratings['ID'] == experiment_tuple[0]]
            else:
                print(f'ERROR: Invalid dataset_type {self._dataset_type}!')
            label_experiment = self._experiment_parameters.loc[self._experiment_parameters['ID'] == experiment_tuple[1]]
            if feature_experiment.shape != expected_input_shape or label_experiment.shape != expected_output_shape:
                self._contiguous_experiments.remove(experiment_tuple)

    def get_experiment_params_by_id(self, experiment_id: int) -> DataFrame:
        """
        selects experiment parameters by experiment_id
        :param experiment_id: experiment identifier
        :return: experiment parameters
        """
        experiment = self._experiment_parameters.loc[self._dataframe['ID'] == experiment_id]
        experiment_without_id = experiment.drop(['ID'], axis=1)
        return experiment_without_id

    def get_experiment_ratings_by_id(self, experiment_id: int) -> DataFrame:
        """
        selects experiment ratings by experiment_id
        :param experiment_id: experiment identifier
        :return: experiment ratings
        """
        experiment = self._experiment_ratings.loc[self._dataframe['ID'] == experiment_id]
        experiment_ratings_without_id = experiment.drop(['ID'], axis=1)
        return experiment_ratings_without_id
