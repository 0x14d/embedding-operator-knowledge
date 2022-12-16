""" schemas: pydantic Models"""

# pylint: disable=import-error

import uuid
from typing import List, Optional, Dict, Union
from enum import Enum

from pydantic import BaseModel, validator, root_validator # pylint: disable=no-name-in-module

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorConfig, KnowledgeGraphGeneratorType
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig

class ModelTypeEnum(str, Enum):
    fnn = 'fnn'
    sss = 'sss'
    ckl = 'ckl'
    cn = 'cn'
    rbl = 'rbl'

class OptimizerConfig(BaseModel):
    name: str
    momentum: float
    weight_decay: float
    learning_rate: float


class FNNConfig(BaseModel):
    nodes_l1: int
    batch_size: int
    epochs: int
    early_stopping: Optional[bool]
    optimizer: OptimizerConfig
    dropout_l1: Optional[float] = 0.5
    dropout_l2: Optional[float] = 0.1


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
    main_folder: str
    output_folder: Optional[str]
    dataset_folder: Optional[str]
    embedding_folder: Optional[str]
    dataset_type: str
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
        assert (sdg is None) ^ (aipe is None), 'Either sdg_config or aipe_kg_config must be provided'
        return values

    @property
    def kg_representation(self) -> KnowledgeGraphGeneratorType:
        if isinstance(self.sdg_config, str):
            self.sdg_config  = SdgConfig.create_config(self.sdg_config)
        if self.sdg_config is not None:
            return self.sdg_config.knowledge_graph_generator.type
        elif self.aipe_kg_config is not None:
            return self.aipe_kg_config.type

class ValidationResult(BaseModel):
    mse: float
    rmse: Optional[float]
    mae: Optional[float]
    r2s: Optional[float]

class ContextNetValidationResult(ValidationResult):
    embedding_loss: float
    combined_loss: float

class RepresentationBasedLossfunctionValidationResult(ValidationResult):
    target_loss: float
    sub_kg_loss: float

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
