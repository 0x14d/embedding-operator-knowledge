""" Information Export class """
import os.path
from textwrap import indent
from typing import List

import numpy as np
import torch
import uuid
from uuid import UUID

from .schemas import EvaluationResult, TrainConfig, FoldResult, AvgResult, ValidationResult


class InformationExport:
    """
    class for exporting training configuration and training result
    """

    def __init__(self, t: str, save_folder: str):
        self._train_id: UUID = uuid.uuid4()
        self._time: str = t
        self._save_folder: str = save_folder

        self._create_folder()

    def _create_folder(self) -> None:
        """
        :return: None
        """
        if not os.path.isdir(self._save_folder):
            os.makedirs(self._save_folder)

    def export_train_configuration(self, config: TrainConfig) -> None:
        """
        export training configuration
        :param config: training configuration
        :return: None
        """
        if config.id is None:
            config.id = self._train_id
        if config.time is None:
            config.time = self._time
        with open(self._save_folder + '/config_' + self._time + '.json', 'w') as out_file:
            config_json_str: str = config.json(indent=4)
            out_file.write(config_json_str)

    def export_evaluation_result(self, results: dict, test_result: ValidationResult) -> None:
        """
        :param test_loss:
        :param results:
        :return: None
        """
        train_loss_sum, train_quality_sum, val_loss_sum, val_quality_sum, time_sum = 0.0, 0.0, 0.0, 0.0, 0.0
        fold_results: List[FoldResult] = []
        np.array([results[i][0].mse for i in range(len(results))])
        train_avg_mse = np.array([results[i][0].mse for i in range(len(results))]).mean()
        train_avg_rmse = np.array([results[i][0].rmse for i in range(len(results))]).mean()
        train_avg_mae = np.array([results[i][0].mae for i in range(len(results))]).mean()
        train_avg_r2s = np.array([results[i][0].r2s for i in range(len(results))]).mean()
        val_avg_mse = np.array([results[i][1].mse for i in range(len(results))]).mean()
        val_avg_rmse = np.array([results[i][1].rmse for i in range(len(results))]).mean()
        val_avg_mae = np.array([results[i][1].mae for i in range(len(results))]).mean()
        val_avg_r2s = np.array([results[i][1].r2s for i in range(len(results))]).mean()
        time_avg = np.array([results[i][2] for i in range(len(results))]).mean()
        for _, value in results.items():
            fold_results.append(
                FoldResult(train_result=value[0], validation_result=value[1], time_needed=value[2]))

        print(f'Averages: Train MSE: {train_avg_mse}, Validation MSE:{val_avg_mse}')
        evaluation_result = EvaluationResult(
            folds=fold_results,
            avg_result=AvgResult(
                train_avg=ValidationResult(mse=train_avg_mse, rmse=train_avg_rmse, mae=train_avg_mae,
                                           r2s=train_avg_r2s),
                validation_avg=ValidationResult(mse=val_avg_mse, rmse=val_avg_rmse, mae=val_avg_mae, r2s=val_avg_r2s),
                time_avg=time_avg
            ),
            test_result=test_result
        )
        with open(self._save_folder + '/result_' + self._time + '.json', 'wt') as out_file:
            config_json_str: str = evaluation_result.json(indent=4)
            out_file.write(config_json_str)

    def export_model_state_dict(self, state_dict: dict) -> None:
        """
        export state dictionary from training model
        :param state_dict:
        :return: None
        """
        save_path: str = self._save_folder + '/model_' + self._time + '.pth'
        torch.save(state_dict, save_path)
