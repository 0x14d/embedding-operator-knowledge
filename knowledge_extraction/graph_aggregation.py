from __future__ import annotations

import copy
from typing import Iterable, Mapping, NamedTuple, Union, List, Dict

from igraph import Graph

import experiment_definition as ed
import knowledge_graph as kg
from knowledge_graph import update_experiment_series
from influences_visualisation import InfluenceVisualiser


class WeightedGraph(NamedTuple):
    graphs: List[Graph]  # multiple graph in case of WeightDuringAggregation
    weights_list: List[float]  # only set during WeightDuringAggregation
    weighted_by: Union[ed.WeightDuringAggregation, ed.WeightingClustering, ed.WeightingInfluenceValidity,
                       ed.DoubleWeightingClustering]


class FilteredGraph(NamedTuple):
    graphs: List[Graph]
    experiment_series: Dict
    weighted_by: Union[ed.WeightDuringAggregation, ed.WeightingClustering, ed.WeightingInfluenceValidity,
                       ed.DoubleWeightingClustering]
    filtered_by: ed.AdaptiveFilters


class GraphAggregation:

    @staticmethod
    def _weight_graph(
            graphs: Iterable[Graph],
            weight_functions: Iterable[ed.WeightingMethod],
            experiment_series: Mapping | None = None,
            edges_dict: Mapping | None = None) -> List[WeightedGraph]:
        """
        weight aggregated or sub graphs depending on the weighting function
        :param graphs <list> of unweighted sub-graphs
        :param weight_functions <list> that contains either members of ExperimentDefinition.WeightDuringAggregation,
        ExperimentDefinition.WeightClustering or ExperimentDefinition.WeightingInfluenceValidity
        :param experiment_series <dict> with experiment series
        :param edges_dict only used for weighting methods ExperimentDefinition.WeightDuringAggregation
        :return tuple of graphs, list of weights and weighting_function
        """
        # dictionary containing tuple of graph and weights list
        weighted_graphs: List[WeightedGraph] = []
        for idx, function in enumerate(weight_functions):
            graph_copy = [copy.deepcopy(graph) for graph in graphs]
            if isinstance(function, ed.WeightDuringAggregation):
                _weighted_graphs, list_weights = kg.weight_graphs(graph_copy, function, experiment_series, edges_dict)
                weighted_graphs.append(WeightedGraph(_weighted_graphs, list_weights, function))
            elif isinstance(function, ed.WeightingClustering):
                graph = kg.aggregate_unfiltered(graph_copy)
                weighted_graph = [InfluenceVisualiser.weight_clustering(graph, experiment_series, function)]
                weighted_graphs.append(WeightedGraph(weighted_graph, [], function))
            elif isinstance(function, ed.DoubleWeightingClustering):
                graph = kg.aggregate_unfiltered(graph_copy)
                weighted_graph = [InfluenceVisualiser.weighting_double_clustering(graph, experiment_series, function)]
                weighted_graphs.append(WeightedGraph(weighted_graph, [], function))
            elif isinstance(function, ed.WeightingInfluenceValidity):
                graph = kg.aggregate_unfiltered(graph_copy)
                weighted_graph = [InfluenceVisualiser.weighting_relations_level(graph, experiment_series, function)]
                weighted_graphs.append(WeightedGraph(weighted_graph, [], function))
            else:
                raise NotImplementedError(f'Unknown type of weight function {function.name}')
        return weighted_graphs

    @staticmethod
    def _filter_graph(weighted_graphs: Iterable[WeightedGraph],
                      filter_functions: Iterable[ed.AdaptiveFilters],
                      experiment_series: dict) -> List[FilteredGraph]:
        """
        filters relations that should not be considered, depending on the weighting function (during aggregation of with
        clustering)
        :param weighted_graphs <list> that either contains an already aggregated graph or a list of weighted sub-graphs
        :param filter_functions <list> filter functions
        :param experiment_series <dict> experiment series
        """
        filtered_graphs: List[FilteredGraph] = []
        for filter_function in filter_functions:
            for weighted_graph in weighted_graphs:
                graphs = weighted_graph.graphs
                weights = weighted_graph.weights_list
                if isinstance(weighted_graph.weighted_by, ed.WeightDuringAggregation):
                    _filtered_graphs = kg.filter_graphs(graphs, weights, filter_function)
                    exp_series = update_experiment_series(_filtered_graphs, copy.deepcopy(experiment_series))
                    filtered_graphs.append(
                        FilteredGraph(kg.aggregate_unfiltered(_filtered_graphs), exp_series, weighted_graph.weighted_by,
                                      filter_function)
                    )
                elif isinstance(weighted_graph.weighted_by, (ed.WeightingClustering, ed.WeightingInfluenceValidity,
                                                             ed.DoubleWeightingClustering)):
                    filtered_graphs.append(
                        FilteredGraph(InfluenceVisualiser.filter_graph(graphs[0], filter_function), experiment_series,
                                      weighted_graph.weighted_by, filter_function)
                    )
                else:
                    raise NotImplementedError(f'Unknown type of filter function {filter_function.name}')

        return filtered_graphs

    @staticmethod
    def get_aggregated_graphs(graphs: Iterable[Graph],
                              weight_functions: Iterable[ed.WeightingMethod],
                              filter_functions: Iterable[ed.AdaptiveFilters],
                              experiment_series: dict, edges_dict: Mapping):
        """
        todo
        """
        weighted_graph_tuples: List[WeightedGraph] = GraphAggregation \
            ._weight_graph(graphs, weight_functions, experiment_series, edges_dict)

        filtered_graphs: List[FilteredGraph] = GraphAggregation._filter_graph(
            weighted_graph_tuples, filter_functions, experiment_series)

        filtered_graphs_dict = dict()
        for filtered_graph in filtered_graphs:
            method = f'{filtered_graph.weighted_by.name.lower()}#{filtered_graph.filtered_by.name.lower()}'
            filtered_graphs_dict[method] = dict()
            filtered_graphs_dict[method]['graph'] = filtered_graph.graphs
            filtered_graphs_dict[method]['exp_series'] = filtered_graph.experiment_series
        return filtered_graphs_dict


if __name__ == '__main__':
    dp = data_provider_singleton.get_data_provider('local')
    data, lok, lov, boolean_parameters, returned_graphs, experiment_series_main, _ = \
        dp.get_experiments_with_graphs(influential_influences_only=True)

    weight_methods = [ed.WeightDuringAggregation.CONFIDENCE_FREQUENCY, ed.WeightDuringAggregation.FREQUENCY]
    filter_methods = [ed.AdaptiveFilters.MEAN]
    aggregated_graphs = GraphAggregation.get_aggregated_graphs(returned_graphs, weight_methods, filter_methods,
                                                               experiment_series_main, dp.get_edges_dict())
