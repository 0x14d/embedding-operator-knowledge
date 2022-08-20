"""
This module provides the class "QuantifiedConditionsKnowledgeGraphGenerator
"""
from typing import List
from math import nextafter, inf
from igraph import Graph
from numpy.random import default_rng as rng
from numpy.random import Generator
import numpy as np

from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.abstract_knowledge_graph_generator import \
    KnowledgeGraphGenerator
from data_provider.synthetic_data_generation.types.generator_arguments import KnowledgeGraphGeneratorArguments
from data_provider.synthetic_data_generation.config.modules.knowledge_graph_generator_config \
    import QuantifiedConditionsKnowledgeGraphGeneratorConfig
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple


class QuantifiedConditionsKnowledgeGraphGenerator(KnowledgeGraphGenerator):
    """
    Class that provides functionality to generate knowledge graphs that display
    a quantified parameter-quality relation
    """
    _config: QuantifiedConditionsKnowledgeGraphGeneratorConfig
    _rng: Generator
    _expert_knowledge: List[PQTuple]
    _pq_functions: GeneratedPQFunctions
    _sdg_config: SdgConfig

    def __init__(self, args: KnowledgeGraphGeneratorArguments) -> None:
        super().__init__()
        self._expert_knowledge = args.pq_tuples.expert_knowledge
        self._config = args.sdg_config.knowledge_graph_generator
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
        # Add parameters to the graph
        process_parameters = sorted(list(set(x[0] for x in pq_tuples_included)))
        process_parameters_attributes = {'type': ['parameter'] * len(process_parameters)}

        knowledge_graph.add_vertices(process_parameters, process_parameters_attributes)

        # Add qualities to the graph
        quality_characteristics = sorted(list(set(x[1] for x in pq_tuples_included)))
        quality_characteristics_attriubutes = {'type': ['qual_influence'] * len(quality_characteristics)}
        knowledge_graph.add_vertices(quality_characteristics, quality_characteristics_attriubutes)

        for pair in pq_tuples_included:
            parameter = pair[0]
            quality = pair[1]

            # Create the bins and sample the value of the function in the middle of the bins
            sample_points = self._create_relation(parameter, quality)
            # sample_points = self._sample_relation(parameter, quality)

            for point in sample_points:

                # Add sampled relation to the graph
                value = str(point['value'])
                bin = point['name']
                bin_start = point['start']
                bin_end = point['end']

                knowledge_graph.add_vertex(value)
                # Knew vertices are appended to a list, get the end of the list to get the vertex object
                value_vertex = knowledge_graph.vs.select()[-1]
                knowledge_graph.add_vertex(bin)
                bin_vertex = knowledge_graph.vs.select()[-1]
                knowledge_graph.add_vertex(bin_start)
                start_vertex = knowledge_graph.vs.select()[-1]
                knowledge_graph.add_vertex(bin_end)
                end_vertex = knowledge_graph.vs.select()[-1]

                # Add edges between the added vertices
                added = knowledge_graph.add_edge(bin_vertex, value_vertex)
                added["weight"] = "implies"
                added["literal_included"] = "To"
                added = knowledge_graph.add_edge(value_vertex, parameter)
                added["weight"] = "quantifies"
                added["literal_included"] = "From"
                added = knowledge_graph.add_edge(bin_vertex, start_vertex)
                added["weight"] = "starts at"
                added["literal_included"] = "To"
                added = knowledge_graph.add_edge(bin_vertex, end_vertex)
                added["weight"] = "ends at"
                added["literal_included"] = "To"
                added = knowledge_graph.add_edge(bin_vertex, quality)
                added["weight"] = "quantifies"
                added["literal_included"] = "None"
                added = knowledge_graph.add_edge(quality, parameter)
                added["weight"] = "implies unquantified"
                added["literal_included"] = "None"

        return knowledge_graph

    def _create_relation(self, parameter: str, quality: str):
        quality_config = self._sdg_config.get_quality_by_name(quality)
        pq_function = self._pq_functions.pq_functions[(parameter, quality)]

        # Sample the array
        arranged = np.arange(quality_config.min_rating, quality_config.max_rating + 1)
        samples = []
        for element in arranged:
            samples.append(pq_function.inverse(element))

        # Create bins
        edges = np.histogram_bin_edges(samples, bins=self._config.number_of_bins)
        # Q = f(P) (for the bin edges)
        q_at_edges = pq_function(edges)

        results = []
        for i in range(0, len(q_at_edges) - 1):
            bin_value = (edges[i] + edges[i + 1]) / 2
            if i == 0:
                bin_start = str(q_at_edges[i])
            else:
                bin_start = str(nextafter(q_at_edges[i], -inf))
            bin_end = str(q_at_edges[i + 1])

            bin_name = ("µ_" + str(i) + ": " + bin_start + "-" + bin_end)
            results.append({'name': bin_name, 'value': bin_value, 'start': bin_start, 'end': bin_end})

        return results
