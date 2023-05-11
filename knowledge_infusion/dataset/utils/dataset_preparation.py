"""
This module provides functionality to prepare a datset dataframe 
and split in into the process parameters and quality ratings.
"""

# pylint: disable=too-many-locals, too-many-arguments

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_dataframe(
    dataframe: pd.DataFrame,
    info: Dict[str, List[str]],
    boolean_parameters: List[str],
    contiguous_experiments: List[Tuple[int, int]],
    dataset_type: str,
    param_normalizer: MinMaxScaler,
    rating_normalizer: MinMaxScaler,
    fit_normalizer: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    filters the data and splits it into the process parameters and the quality ratings.
    :param dataframe: dataframe containing the dataset
    :param info: infos of the process parameters and quality ratings
    :param boolean_parameters: list of boolean parameters
    :param contiguous_experiments: list of contiguous experiments
    :param dataset_type: type of the dataset
    :param param_normalizer: normalizer for the process parameters
    :param rating_normalizer: normalizer for the quality ratings
    :param fit_normalizer: whether to fit the normalizer before transforming or not
    :return: dataframe containing the process parameters and another containing the quality ratings
    """
    # Process parameters
    process_parameters = [
        process_param for process_param in sorted(list(info['process_parameters']))
        if process_param in dataframe.columns and
        process_param not in boolean_parameters
    ]
    process_params_df = dataframe.loc[:, process_parameters]
    # delete columns with string
    for column in process_params_df:
        element = process_params_df.loc[0, column]
        if isinstance(element, str):
            process_params_df.drop([column], axis=1, inplace=True)

    # Ratings
    ratings = sorted(list(info['ratings']))
    ratings.remove('overall_ok')
    ratings_df = dataframe.loc[:, ratings]

    # Normalize
    if fit_normalizer:
        param_normalizer.fit(process_params_df.to_numpy())
        rating_normalizer.fit(ratings_df.to_numpy())

    process_params_df = pd.DataFrame(
        param_normalizer.transform(process_params_df),
        columns=process_params_df.columns
    )
    ratings_df = pd.DataFrame(
        rating_normalizer.transform(ratings_df),
        columns=ratings_df.columns
    )

    # Add ID
    process_params_df['ID'] = dataframe.loc[:, dataframe.columns == 'ID']
    ratings_df['ID'] = dataframe.loc[:, dataframe.columns == 'ID']

    # Check completeness

    feature_experiment: Optional[pd.DataFrame] = None
    expected_input_shape: Optional[Tuple[int, int]] = None
    expected_output_shape: Tuple[int, int] = (
        1, 47)  # TODO: Remove magic numbers
    for experiment_tuple in contiguous_experiments:
        if dataset_type == 'param2param':
            expected_input_shape = (1, 47)  # TODO: Remove magic numbers
            feature_experiment = process_params_df.loc[
                process_params_df['ID'] == experiment_tuple[0]
            ]
        # TODO refactor check - here only ratings are compared input shape is a suboptimal name
        elif dataset_type in ['ratings2param', 'ratings&param2param']:
            expected_input_shape = (1, 14)  # TODO: Remove magic numbers
            feature_experiment = ratings_df.loc[ratings_df['ID']
                                                == experiment_tuple[0]]
        else:
            print(f'ERROR: Invalid dataset_type {dataset_type}!')
        label_experiment = process_params_df.loc[process_params_df['ID']
                                                 == experiment_tuple[1]]
        if feature_experiment.shape != expected_input_shape or \
                label_experiment.shape != expected_output_shape:
            contiguous_experiments.remove(experiment_tuple)

    return process_params_df, ratings_df
