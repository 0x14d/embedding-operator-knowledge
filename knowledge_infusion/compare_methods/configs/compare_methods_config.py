"""This module defines the a class that can be used to configure compare_methods.py
it provides an abstract class from which a concret configuration class can
inherit.
Additionally three example configuration classes are provided, which should be
sufficient for most uses currently planned. (AMRI & Hits@k Evaluation,
Matches@k Evaluation, Debugging of the Evaluation)
"""

# pylint: disable=import-error

from abc import ABC
from enum import Enum

from rule_base.rule_extraction import BinnedInfluences
from knowledge_infusion.graph_embeddings.embedding_types import EmbeddingType
from knowledge_infusion.graph_embeddings.embedding_config import \
    EmbeddingConfig, NormalizationMethods, StandardConfig, KnowledgeExtractionMethod


class EvaluationMethod(Enum):
    AmriAndHitsAtK = 1
    MatchesAtK = 2


class DataProvider(str, Enum):
    SYNTHETIC = 'synthetic'
    AIPE = 'fdm'


class CompareMethodsConfig(ABC):
    """Class used to configure CompareMethods
    """
    _name: str
    evaluationMethod: EvaluationMethod
    embedding_types: list[str]
    head_vals: list[bool]
    distance_metrics: list[str]
    generate_rdf_graph: bool
    generate_graph_infos: bool
    times: int
    embedding_config: EmbeddingConfig
    sdg_config: str
    train_config: str
    results_folder: str
    data_provider: DataProvider

    @property
    def name(self) -> str:
        """Name of the config"""
        return f'{self._name}/{self.data_provider.name}/' + \
               f'{self.embedding_config.rule_extraction_method.mode.name}/' + \
               f'{self.embedding_config.knowledge_extraction_method}'

    def __init__(self,
                 name: str,
                 evaluation_method: EvaluationMethod,
                 embedding_types: list[str],
                 head_vals: list[bool],
                 distance_metrics: list[str],
                 generate_rdf_graph: bool,
                 generate_graph_infos: bool,
                 times: int,
                 embedding_config: int,
                 data_provider: DataProvider
                 ) -> None:
        super().__init__()
        self._name = name
        self.evaluation_method = evaluation_method
        self.embedding_types = embedding_types
        self.head_vals = head_vals
        self.distance_metrics = distance_metrics
        self.generate_rdf_graph = generate_rdf_graph
        self.generate_graph_infos = generate_graph_infos
        self.times = times
        self.embedding_config = embedding_config
        self.data_provider = data_provider

        self.sdg_config = 'knowledge_infusion/hyperparameter_tuning/configs/default_config_sdg.json'
        self.train_config = 'knowledge_infusion/hyperparameter_tuning/configs/equal_config_fnn.json'
        self.results_folder = "knowledge_infusion/compare_methods/results/"

    def save_configuration(self, file_location: str) -> None:
        """Saves this configuration to a given location

        Args:
            file_location (str): Where the configuration should be saved
        """

        details = []

        details.append(f'Name:                  {self.name}\n')
        details.append(f'Evaluation Method:     {self.evaluation_method}\n')
        details.append(f'Head Values:           {self.head_vals}\n')
        details.append(f'Distance Metrics:      {self.distance_metrics}\n')
        details.append(f'TIMES:                 {self.times}\n')
        details.append(f'Data Provider:         {self.data_provider}\n')
        details.append(f'Embedding Types:\n')

        for embedding in self.embedding_types:
            details.append(f'                       {embedding},\n')

        with open(file_location, 'w') as out_f:
            out_f.writelines(details)

        return details


class AmriConfig(CompareMethodsConfig):
    """This config is used for generating link prediction results.
    """

    def __init__(self) -> None:
        super().__init__(
            evaluation_method=EvaluationMethod.AmriAndHitsAtK,
            embedding_types=[
                EmbeddingType.TransE,
                EmbeddingType.ComplEx,
                EmbeddingType.ComplExLiteral,
                EmbeddingType.RotatE,
                EmbeddingType.DistMult,
                EmbeddingType.DistMultLiteralGated,
                EmbeddingType.BoxE,
                EmbeddingType.Rdf2Vec
            ],
            head_vals=[True],
            distance_metrics=["euclidean", "jaccard"],
            generate_rdf_graph=False,
            generate_graph_infos=True,
            times=1,
            embedding_config=StandardConfig(),
            data_provider=DataProvider.AIPE,
            name="LinkPrediction")


class MatchesAtKConfig(CompareMethodsConfig):
    """This config should be used for evaluating the knowledge graphs with the
    matches@k metrik
    """

    def __init__(self) -> None:
        super().__init__(
            evaluation_method=EvaluationMethod.MatchesAtK,
            embedding_types=[
                EmbeddingType.TransE,
                EmbeddingType.ComplEx,
                EmbeddingType.ComplExLiteral,
                EmbeddingType.RotatE,
                EmbeddingType.DistMult,
                EmbeddingType.DistMultLiteralGated,
                EmbeddingType.BoxE,
                EmbeddingType.Rdf2Vec
            ],
            head_vals=[True, False],
            distance_metrics=["euclidean", "jaccard"],
            generate_rdf_graph=False,
            generate_graph_infos=False,
            times=1,
            embedding_config=StandardConfig(),
            data_provider=DataProvider.AIPE,
            name="Matches")


class DebugConfig(CompareMethodsConfig):
    """This config should be used for debugging the code
    """

    def __init__(self) -> None:
        super().__init__(
            name="debug",
            evaluation_method=EvaluationMethod.AmriAndHitsAtK,
            embedding_types=[
                EmbeddingType.TransE
            ],
            head_vals=[True],
            distance_metrics=["euclidean"],
            generate_rdf_graph=True,
            generate_graph_infos=True,
            times=1,
            embedding_config=NodeEmbeddingDebugConfig(),
            data_provider=DataProvider.SYNTHETIC)


class NodeEmbeddingDebugConfig(EmbeddingConfig):
    """This is a custom node embedding config for debug purposes and is used
    by the debug config. When using this config each embedding will only train
    for 1 iteration, resulting in quick but very unprecise results
    """

    def __init__(self) -> None:
        super().__init__(epochs={
            EmbeddingType.TransE: 1,
            EmbeddingType.ComplEx: 1,
            EmbeddingType.ComplExLiteral: 1,
            EmbeddingType.RotatE: 1,
            EmbeddingType.DistMult: 1,
            EmbeddingType.DistMultLiteralGated: 1,
            EmbeddingType.BoxE: 1,
            EmbeddingType.Rdf2Vec: 1
        },
            embedding_dim=46,
            train_test_split=0.2,
            rdf2vec_walker_max_depth=4,
            rdf2vec_walker_max_walks=2,
            subkg_normalization_method=NormalizationMethods.number_nodes,
            knowledge_extraction_method=KnowledgeExtractionMethod.AGGREGATE_UNFILTERED,
            knowledge_extraction_filter_function=None,
            knowledge_extraction_weight_function=None,
            rule_extraction_method=BinnedInfluences
            )
