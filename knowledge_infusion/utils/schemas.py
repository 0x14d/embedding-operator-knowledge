""" schemas: pydantic Models"""
# no-self-argument is needed for BaseModel methods
# pylint: disable=import-error,no-self-argument

import uuid
from typing import Any, List, Optional, Dict, Union
from enum import Enum
import warnings
from numpy import ndarray

from pydantic import BaseModel, validator, root_validator, Extra  # pylint: disable=no-name-in-module

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorConfig, KnowledgeGraphGeneratorType
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig


class ModelTypeEnum(str, Enum):
    fnn = 'fnn'
    sss = 'sss'
    ckl = 'ckl'
    cn = 'cn'
    rbl = 'rbl'


class DatasetType(str, Enum):
    PARAM2PARAM = 'param2param'
    RATINGS2PARAM = 'ratings2param'
    RATINGSPARAM2PARAM = 'ratings&param2param'


class OptimizerConfig(BaseModel):
    name: str
    momentum: float
    weight_decay: float
    learning_rate: float


class FNNConfig(BaseModel, extra=Extra.allow):
    nodes: List[int]
    dropouts: List[float]
    batch_size: int
    epochs: int
    early_stopping: Optional[bool]
    optimizer: OptimizerConfig

    @root_validator(pre=True)
    def legacy_structure(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Allows the usage of the old fnn architecture config"""
        if 'nodes_l1' in values:
            if 'nodes' not in values:
                values['nodes'] = [values['nodes_l1']]
            else:
                warnings.warn(
                    '[FNNConfig] nodes_l1 will be ignored cause nodes was provided')
            del values['nodes_l1']

        if 'dropout_l1' in values and 'dropout_l2' in values:
            if 'dropouts' not in values:
                values['dropouts'] = [
                    values['dropout_l1'], values['dropout_l2']]
            else:
                warnings.warn(
                    '[FNNConfig] dropout_l1 and dropout_l2 will be ignored cause dropouts was provided'
                )
            del values['dropout_l1']
            del values['dropout_l2']

        return values

    @root_validator(pre=True)
    def node_dropout_length(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Checks if the amout of provided nodes and dropouts is valid"""
        if len(values['nodes']) < 1:
            raise ValueError('The lenght of nodes must be at least 1')
        if len(values['dropouts']) < 2:
            raise ValueError('The lenght of dropouts must be at least 2')
        if len(values['nodes']) != len(values['dropouts']) - 1:
            raise ValueError(
                'The length of nodes must equal the length of dropouts - 1')
        return values


class KILConfig(BaseModel):
    nodes: int
    threshold: float
    max_iter: Optional[int]
    optimizer: OptimizerConfig


class CNConfig(BaseModel):
    lambda_target: float
    lambda_embedding: float


class RBLConfig(BaseModel):
    lambda_target: float
    lambda_sub_kg: float


class TrainConfig(BaseModel):
    id: Optional[uuid.UUID]
    time: Optional[str]
    seed: Optional[int]
    iteration_seed: Optional[int]
    main_folder: str
    output_folder: Optional[str]
    dataset_folder: Optional[str]
    embedding_folder: Optional[str]
    dataset_type: DatasetType
    data_provider: Optional[str]
    sdg_config: Optional[Union[str, SdgConfig]]
    model_type: ModelTypeEnum
    fnn_config: FNNConfig
    kil_config: Optional[KILConfig]
    cn_config: Optional[CNConfig]
    rbl_config: Optional[RBLConfig]
    aipe_kg_config: Optional[KnowledgeGraphGeneratorConfig]
    k_fold_size: int
    embedding_method: Optional[str]

    @validator('sdg_config')
    def load_sdg_config(cls, config):
        if isinstance(config, str):
            config = SdgConfig.create_config(config)
        return config

    @validator('dataset_folder')
    def set_dataset_folder(cls, dataset_folder, values):
        if dataset_folder is not None:
            return dataset_folder
        return values['main_folder']

    @validator('embedding_folder')
    def set_embedding_folder(cls, embedding_folder, values):
        if embedding_folder is not None:
            return embedding_folder
        return values['dataset_folder']

    @root_validator(pre=True)
    def check_for_sdg_config_or_aipe_kg_config(cls, values):
        sdg = values.get('sdg_config')
        aipe = values.get('aipe_kg_config')
        assert (sdg is None) ^ (
            aipe is None), 'Either sdg_config or aipe_kg_config must be provided'
        return values

    @property
    def kg_representation(self) -> Optional[KnowledgeGraphGeneratorType]:
        if isinstance(self.sdg_config, str):
            self.sdg_config = SdgConfig.create_config(self.sdg_config)
        if self.sdg_config is not None:
            return self.sdg_config.knowledge_graph_generator.type
        elif self.aipe_kg_config is not None:
            return self.aipe_kg_config.type


class ValidationResult(BaseModel):
    mse: float
    rmse: Optional[float]
    mae: Optional[float]
    r2s: Optional[Union[float, ndarray]]

    class Config:
        arbitrary_types_allowed = True


class ContextNetValidationResult(ValidationResult):
    embedding_loss: float
    combined_loss: float


class RepresentationBasedLossfunctionValidationResult(ValidationResult):
    sub_kg_loss: float
    combined_loss: float


class FoldResult(BaseModel):
    train_result: ValidationResult
    validation_result: ValidationResult
    time_needed: float


class AvgResult(BaseModel):
    train_avg: ValidationResult
    validation_avg: ValidationResult
    time_avg: float


class EvaluationResult(BaseModel):
    folds: List[FoldResult]
    avg_result: AvgResult
    test_result: ValidationResult


class TrainInformation(BaseModel):
    train_config: TrainConfig
    result: EvaluationResult


class ParameterValidationResult(BaseModel):
    mse: float
    recall: float


class ParameterFoldResult(BaseModel):
    train_result: Dict[str, ParameterValidationResult]
    validation_result: Dict[str, ParameterValidationResult]


class ParameterAvgResult(BaseModel):
    train_avg: Dict[str, ParameterValidationResult]
    validation_avg: Dict[str, ParameterValidationResult]


class ParameterEvaluationResult(BaseModel):
    folds: List[ParameterFoldResult]
    avg_result: ParameterAvgResult
