"""
This module provides the class `BasicKnowledgeGraphGenerator`.
"""

# pylint: disable=import-error, missing-function-docstring

import uuid
from typing import List
from igraph import Graph
from numpy import mean
from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.modules.knowledge_graph_generator_config \
    import BasicKnowledgeGraphGeneratorConfig, EdgeWeightCalculationMethod
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.modules.knowledge_graph_generators. \
    abstract_knowledge_graph_generator import KnowledgeGraphGenerator
from data_provider.synthetic_data_generation.types.generator_arguments \
    import KnowledgeGraphGeneratorArguments
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple


class BasicKnowledgeGraphGenerator(KnowledgeGraphGenerator):
    """
    Class that provides functionality to generate knowledge graphs.

    It randomly chooses pq-tuples from the expert knwoledge and adds them to the knowledge graph.
    """

    _config: BasicKnowledgeGraphGeneratorConfig
    _expert_knowledge: List[PQTuple]
    _pq_functions: GeneratedPQFunctions
    _rng: Generator
    _sdg_config: SdgConfig

    def __init__(self, args: KnowledgeGraphGeneratorArguments) -> None:
        super().__init__()

        self._config = args.sdg_config.knowledge_graph_generator
        self._expert_knowledge = args.pq_tuples.expert_knowledge
        self._pq_functions = args.pq_functions
        self._sdg_config = args.sdg_config
        
        self._rng = rng(self._config.seed)

    def generate_knowledge_graph(self) -> Graph:
        knowledge_graph = Graph(directed=True)

        # Choose included expert knowledge
        num_included_tuples = int(len(self._expert_knowledge) * self._config.knowledge_share)
        pq_tuples_included = self._rng.choice(
            a=self._expert_knowledge,
            size=num_included_tuples,
            replace=False
        )

        # Add verticies
        parameters = sorted(list(set(x[0] for x in pq_tuples_included)))
        qualities = sorted(list(set(x[1] for x in pq_tuples_included)))
        parameter_attributes = {'type': ['parameter'] * len(parameters)}
        quality_attributes = {'type': ['qual_influence'] * len(qualities)}
        knowledge_graph.add_vertices(parameters, parameter_attributes)
        knowledge_graph.add_vertices(qualities, quality_attributes)

        # Add edges
        edge_weights = [self._calc_edge_weight(p, q) for p, q in pq_tuples_included]
        qp_tuples_included = [(t[1], t[0]) for t in pq_tuples_included]
        edge_attributes = {
            'weight': edge_weights,
            'experiments': [uuid.uuid1().int >> 64 for _ in pq_tuples_included]
        }
        knowledge_graph.add_edges(qp_tuples_included, edge_attributes)

        return knowledge_graph

    def _calc_edge_weight(self, parameter: str, quality: str) -> float:
        quality_config = self._sdg_config.get_quality_by_name(quality)
        pq_function = self._pq_functions.pq_functions[(parameter, quality)]

        if self._config.edge_weight == EdgeWeightCalculationMethod.MEAN_ABSOLUTE:
            changes = []
            p_old = pq_function.inverse(quality_config.max_rating)
            for q_new in range(quality_config.max_rating - 1, quality_config.min_rating - 1, -1):
                p_new = pq_function.inverse(q_new, last_parameter=p_old)
                changes.append(p_new - p_old)
                p_old = p_new
            return mean(changes)

        raise NotImplementedError(
            f'Missing implementation for edge weight calculation method {self._config.edge_weight}!'
        )
