""" schemas: pydantic Models"""

import uuid
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, validator


class ModelTypeEnum(str, Enum):
    fnn = 'fnn'
    sss = 'sss'
    ckl = 'ckl'


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


class KILConfig(BaseModel):
    nodes: int
    threshold: float
    max_iter: Optional[int]
    embedding_folder: Optional[str]
    optimizer: OptimizerConfig


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
    sdg_config: Optional[str]
    model_type: ModelTypeEnum
    fnn_config: FNNConfig
    kil_config: KILConfig
    k_fold_size: int

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

class ValidationResult(BaseModel):
    mse: float
    rmse: Optional[float]
    mae: Optional[float]
    r2s: Optional[float]


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
