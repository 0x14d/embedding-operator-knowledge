""" schemas: pydantic Models"""

import uuid
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel


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
    embedding_folder: str
    optimizer: OptimizerConfig


class TrainConfig(BaseModel):
    id: Optional[uuid.UUID]
    time: Optional[str]
    seed: Optional[int]
    main_folder: str
    dataset_type: str
    model_type: ModelTypeEnum
    fnn_config: FNNConfig
    kil_config: KILConfig
    k_fold_size: int


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
