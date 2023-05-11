# pylint: disable=import-error

from abc import ABC
from enum import Enum
from typing import Optional

from experiment_definition import WeightingMethod, AdaptiveFilters
from knowledge_infusion.graph_embeddings.embedding_types import EmbeddingType
from rule_base.rule_extraction import RuleExtractionMethod, FromEdge


class NormalizationMethods(str, Enum):
    unit_norm = "unit_norm"
    number_nodes = "num_nodes"
    none = "none"


class KnowledgeExtractionMethod(str, Enum):
    AGGREGATE_UNFILTERED = 'aggregate_unfiltered'
    AGGREGATE_FILTERED = 'aggregate_filtered'
    GROUNDTRUTH = 'groundtruth'


class EmbeddingConfig(ABC):
    """
    This class can be given to the Embedding class to configure non standard
    behaviour.
    """
    epochs: dict
    embedding_dim: int

    train_test_split: float

    rdf2vec_walker_max_depth: int
    rdf2vec_walker_max_walks: int

    subkg_normalization_method: NormalizationMethods
    knowledge_extraction_method: KnowledgeExtractionMethod
    knowledge_extraction_weight_function: Optional[WeightingMethod]
    knowledge_extraction_filter_function: Optional[AdaptiveFilters]
    rule_extraction_method: RuleExtractionMethod

    def __init__(
        self,
        epochs: dict,
        embedding_dim: int,
        train_test_split: float,
        rdf2vec_walker_max_depth: int,
        rdf2vec_walker_max_walks: int,
        subkg_normalization_method: NormalizationMethods,
        knowledge_extraction_method: KnowledgeExtractionMethod
            = KnowledgeExtractionMethod.AGGREGATE_UNFILTERED,
        knowledge_extraction_weight_function: Optional[WeightingMethod] = None,
        knowledge_extraction_filter_function: Optional[AdaptiveFilters] = None,
        rule_extraction_method: RuleExtractionMethod = FromEdge
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.train_test_split = train_test_split
        self.rdf2vec_walker_max_depth = rdf2vec_walker_max_depth
        self.rdf2vec_walker_max_walks = rdf2vec_walker_max_walks
        self.subkg_normalization_method = subkg_normalization_method
        self.knowledge_extraction_method = knowledge_extraction_method
        self.knowledge_extraction_weight_function = knowledge_extraction_weight_function
        self.knowledge_extraction_filter_function = knowledge_extraction_filter_function
        self.rule_extraction_method = rule_extraction_method

        if knowledge_extraction_method == KnowledgeExtractionMethod.AGGREGATE_FILTERED and \
           not all([knowledge_extraction_weight_function, knowledge_extraction_filter_function]):
            raise ValueError(
                'Weight and filter function must be defined when using filtered aggregation!')
        if knowledge_extraction_method != KnowledgeExtractionMethod.AGGREGATE_FILTERED and \
           any([knowledge_extraction_weight_function, knowledge_extraction_filter_function]):
            raise ValueError(
                'Weight and filter function mustn\'t be defined when not using filtered aggregation!')


class StandardConfig(EmbeddingConfig):
    def __init__(self) -> None:
        super().__init__(
            epochs={
                EmbeddingType.TransE: 400,
                EmbeddingType.ComplEx: 1000,
                EmbeddingType.ComplExLiteral: 650,
                EmbeddingType.RotatE: 700,
                EmbeddingType.DistMult: 800,
                EmbeddingType.DistMultLiteralGated: 200,
                EmbeddingType.BoxE: 1500,
                EmbeddingType.Rdf2Vec: 1000
            },
            embedding_dim=46,
            train_test_split=0.2,
            rdf2vec_walker_max_depth=4,
            rdf2vec_walker_max_walks=100,
            subkg_normalization_method=NormalizationMethods.number_nodes,
            knowledge_extraction_method=KnowledgeExtractionMethod.AGGREGATE_UNFILTERED,
            knowledge_extraction_weight_function=None,
            knowledge_extraction_filter_function=None,
            rule_extraction_method=FromEdge
        )
