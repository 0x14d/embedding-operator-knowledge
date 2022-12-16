from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import operator
from collections import defaultdict
from distutils import util
from functools import partial
from typing import Callable, Iterable

import deprecation
import igraph
import numpy as np
import pandas
from data_provider import abstract_data_provider

from preprocessing import Preprocessing, LabelEncoderForColumns



def __add_vertices_for_type(graph, list_of_vertices_dicts, type_of_vertex=None):
    """
    Adds len(dict_of_attributes) vertices to the graph
    :param list_of_vertices_dicts: a dictionary with attributes for each vertex to ad
    :param type_of_vertex: the type of the vertex to add currently we have type \in {'env_influence', 'qual_influence', 'parameter'}
    :return:
    """
    for vertex in list_of_vertices_dicts:
        # create a vertex for each parameter and set its values
        vertex['type'] = type_of_vertex
        graph.add_vertex(vertex['key'], **vertex)


def __influences_to_flat_array(insight_items, **kwargs):
    """
    Flattens a insight_item returning the corresponding key value dict
    :param insight_items:
    :param kwargs: optional arguments to add to the dict
    :return:
    """
    a = ('key', insight_items[0])
    b = ('value', insight_items[1]['value'])
    dict_init = [a, b]
    if kwargs:
        # concatenate dict_init with additional arguments if neccessary
        dict_init += list(kwargs.items())
    return dict(dict_init)


def __transform_insights(experiment, label_encoder):
    """parses the relevant areas of the experiment to proper values"""
    # TODO XAI-587 should we transofrm experiment[all_values] as well?
    for parameter in experiment['insights']['changed_ui_parameters']:
        for dict_key in ['user_value', 'original_value']:
            try:
                parameter[dict_key] = float(parameter[dict_key])
            except ValueError:
                try:
                    parameter[dict_key] = float(
                        util.strtobool(parameter[dict_key]))
                except ValueError:
                    # we encountered an unfloatable value -> use label_encoder to determine label encoded value
                    parameter[dict_key] = label_encoder.transform(
                        parameter[dict_key].lower().replace(" ", ""),
                        parameter['key'])
    for influence in experiment['insights']['influences']:
        for key, value in influence.items():
            try:
                value['value'] = float(value['value'])
            except ValueError:
                try:
                    value['value'] = float(util.strtobool(value['value']))
                except ValueError:
                    # we encountered an unfloatable value -> use label_encoder to determine label encoded value
                    value['value'] = label_encoder.transform(
                        value['value'].lower().replace(" ", ""), key)


def parse_insights_to_graph(experiment, label_encoder: LabelEncoderForColumns, influential_only=True,
                            edges_dict=None):
    """Parses given insights to a graph that can be aggregated later on"""
    # For issues with igraph imports, please check readme.md for an
    # instruction
    if edges_dict is None:
        edges_dict = dict()
    import copy
    experiment = copy.deepcopy(experiment)
    __transform_insights(experiment, label_encoder)
    from igraph import Graph
    g = Graph(directed=True)
    # vertices
    insights = experiment['insights']
    parameters = insights['changed_ui_parameters']
    __add_vertices_for_type(g, parameters, 'parameter')
    # format & filter influences
    influences = insights['influences']
    environment_influences = {}
    for influence in influences:
        environment_influences.update(influence)
    if influential_only:
        influential = dict(filter(lambda elem: elem[1]['influential'] is True, environment_influences.items()))
    else:
        influential = environment_influences
    flat_influential = list(map(__influences_to_flat_array, influential.items()))
    __add_vertices_for_type(g, flat_influential, 'env_influence')

    flat_prev_qual = []
    influenced_by = None
    influential_qual = {}
    if 'last_rating_influences' in insights:
        if insights['last_rating_influences'][0] is not None:
            # format & filter quality ratings
            prev_qual = insights['last_rating_influences']
            prev_qual_per_exp = prev_qual[
                0]  # currently set to 0 since we only have the possibility to select 1 previous experiment
            additional_arguments = {'rated_id': prev_qual_per_exp['experiment_id']}
            influenced_by = prev_qual_per_exp['experiment_id']
            del prev_qual_per_exp['experiment_id']
            del prev_qual_per_exp['comment']
            prev_qual_per_exp = {key: value for (key, value) in prev_qual_per_exp.items()
                                 if type(key) != int and value['value'] != 0 and value['value'] != False}
            if influential_only:
                influential_qual = dict(filter(lambda elem: elem[1]['influential'] is True, prev_qual_per_exp.items()))
            else:
                influential_qual = prev_qual_per_exp
            flat_prev_qual = list(
                map(lambda x: __influences_to_flat_array(x, **additional_arguments), influential_qual.items()))
            __add_vertices_for_type(g, flat_prev_qual, 'qual_influence')

    influences_keys = list(map(lambda x: x['key'], flat_prev_qual + flat_influential))
    parameters_keys = list(map(lambda x: x['key'], parameters))
    # TODO: even if experiments with insights are considered here, they might have no influences specified. Consequently, there
    #  will be no edges in the graph and as we are weighting the edges the graph will have no weight => XAI-574

    edges = itertools.product(influences_keys, parameters_keys)
    g.add_edges(edges)

    edges_in_graph = get_edges(g)
    for edge in edges_in_graph:
        key = json.dumps(edge)
        if key in edges_dict.keys():
            edges_dict[key] += 1
        else:
            edges_dict[key] = 1

    # give the edges also a type so that we know from which type of vertex they are coming from
    for edge in g.es:
        if edge.source_vertex['type'] == 'qual_influence':
            edge['type'] = 'qual_influence'
        elif edge.source_vertex['type'] == 'env_influence':
            edge['type'] = 'env_influence'

    if 'uncertainty' in insights:
        g.es["weight"] = insights['uncertainty']
    else:
        g.es['weight'] = 0.5
    # keep track of which edge was created by which experiment
    g.es['experiment'] = experiment['_id']
    g['experiment_id'] = experiment['_id']
    g['environment'] = abstract_data_provider.AbstractDataProvider.aggregate(experiment['measurements'], 'median')
    g['quality'] = abstract_data_provider.AbstractDataProvider.aggregate(experiment['ratings'], 'mean')
    import pandas
    g['quality_aggregated'] = Preprocessing.transform_rating_linear(pandas.DataFrame.from_dict(
        {setting_name: [setting_value] for setting_name, setting_value in g['quality'].items()}))['rating'].iloc[0]
    g['environment_influences'] = environment_influences
    g['all_parameters'] = experiment['all_parameters']
    g['influenced_by'] = influenced_by
    g['quality_influences'] = influential_qual
    return g


def get_edges(graph):
    edges = list()
    for edge in graph.es:
        edges.append((edge.source_vertex['name'], edge.target_vertex['name']))
    return edges


def weight_graphs(graphs, weighting_function,
                  experiment_series=None, edges_dict=None):
    """
    Weight all the graphs available in the given list
    You can combine different weight strategies to consider certain methods
    Graphs are pre-weighted with the confidence or a default value. This weight will be either considered if
    with_confidence is true or the weight will be replaced
    :param graphs list of all graphs to be weighted
    :param weighting_function function to weight the single graphs
    :param experiment_series list with experiment series to show which experiments have a relation with each other
    :return weighted graphs and a list with all used weights. The list of graphs may not have the same length as the
    list of used weights because some graphs do not have edges so you cannot weight the graph
    """
    # ensure that we do not modify data that should be immutable
    if experiment_series is None:
        experiment_series = list()
    if edges_dict is None:
        edges_dict = dict()
    new_graphs = [copy.deepcopy(graph) for graph in graphs]
    list_of_all_weights = list()
    # TODO check if the quality improvement relates to the previous experiment
    knowledge_graph = aggregate_unfiltered(new_graphs)  # aggregate the current graph
    es_ps = extract_e_p_values(knowledge_graph, experiment_series)
    aggregated_improvement = dict()  # dict that contains for every experiment id the improvement of all ratings
    influence_considered = dict()  # dict that contains every rating for each exp_id that has already been detected
    for e_p in es_ps:
        df = es_ps[e_p]
        parameter, influence = e_p.split('-')
        try:
            # TODO excessively complicated and could cause errors when run with environmental influences -> get quality_realtive df and sum each rows
            df[f'relative_{influence}'] = -1 * (df[influence] - df['influence_value'])  # calculate the improvement
            for index, row in df.iterrows():
                exp_id = row['exp_id']
                if exp_id in influence_considered:
                    if influence in influence_considered[exp_id]:  # if influence already considered continue with
                        # next e_p pair
                        continue
                else:
                    influence_considered[exp_id] = list()

                influence_considered[exp_id].append(influence)  # add the considered influence to the dictionary
                # consider the improvement to the last experiment for the current exp_id
                if exp_id in aggregated_improvement.keys():
                    aggregated_improvement[exp_id] += row[f'relative_{influence}']
                else:
                    aggregated_improvement[exp_id] = row[f'relative_{influence}']
        except Exception:
            continue

    # with frequency: number of pairs is the weight
    # with confidence: confidence is the weight
    # combination of frequency+confidence: number of experiments for pair + confidence * number of experiments for pair
    for graph in new_graphs:
        weights = list()
        edges = get_edges(graph)
        # as the graphs are pre-weighted with confidence the value stays the same or will be deleted
        if len(graph.es["weight"]) > 0:
            confidence = graph.es["weight"][0]  # remember the original value for weight
        else:
            confidence = 0.5  # default value if there is no weight for this graph
            weight = confidence

        for edge in edges:
            key = json.dumps(edge)
            frequency = edges_dict[key] / len(edges_dict.items())
            exp_id = graph['experiment_id']  # get the experiment id of the current graph
            if exp_id in aggregated_improvement.keys():
                quality_improvement = aggregated_improvement[exp_id]  # get the improvement of the current experiment
            else:
                quality_improvement = math.nan
            weights.append(weighting_function(confidence, frequency, quality_improvement))
        if len(weights) == 0:
            weights = [weight for _ in range(len(graph.es))]
        graph.es["weight"] = weights
        # the list_of_all_weights can be used later to filter the graphs according to a specific threshold
        # so we do not want to have all calculated weights in this list but these that are added to graph that really
        # has some edges
        if len(graph.es["weight"]) > 0:
            list_of_all_weights.extend(weights)

    filtered_weights = [x for x in list_of_all_weights if not math.isnan(x)]
    return new_graphs, filtered_weights


def weight_aggregated_graph(graph, clusters):
    """
    updates the relation (=edges) weights of graph based on clusters
    :param graph:
    :param clusters: dict mapping experiment_id to cluster_id)
    :return: weighted graph
    """
    transformed_clusters = {exp_id: 1 if value >= 0 else 0
                            for exp_id, value in clusters.items()}
    relations = graph.es
    weighted_graph = graph.copy()
    for relation in relations:
        experiments = relation['experiments']
        weight = 0
        for experiment in experiments:
            try:
                weight += transformed_clusters[experiment]
            except KeyError:
                continue
        weight /= len(experiments)
        # weight /= len(relations)
        weighted_graph.es[relation.index]['weight'] = weight

    return weighted_graph


def filter_graphs(list_of_graphs: Iterable[igraph.Graph], list_of_all_weights: list[float], filter_function: Callable):
    """
    Calculate an adaptive threshold according to the given method and filter graphs that are below the threshold
    :param list_of_graphs: list of weighted graphs
    :param list_of_all_weights: list
    :param filter_function: function to find a threshold in a list of weights
    :return list of filtered graphs
    """
    # ensure that we do not modify data that should be immutable
    new_graphs = [copy.deepcopy(graph) for graph in list_of_graphs]
    all_weights = np.array(list_of_all_weights)
    threshold = filter_function(all_weights)
    filtered_graphs = list()
    for graph in new_graphs:
        edges = get_edges(graph)
        # you need to iterate over the reversed list since otherwise if you delete an edge enumerate still goes to the
        # old length of the edges list and you will get a index out of array
        edge_deleted = False
        for i, edge in reversed(list(enumerate(edges))):
            if graph.es["weight"][i] < threshold:
                graph.delete_edges(edge)
                edge_deleted = True
        if len(graph.es) > 0:
            filtered_graphs.append(graph)
        if edge_deleted is True:
            connected_vertices = set()
            # check which vertices are connected to any edge
            for edge in graph.es:
                connected_vertices.add(edge.source_vertex.index)
                connected_vertices.add(edge.target_vertex.index)
            # delete those vertices that are not connected to any edge
            for vertex in reversed(graph.vs):
                if not (vertex.index in connected_vertices):
                    graph.delete_vertices(vertex)

    return filtered_graphs


def filter_aggregated_graph(graph, threshold_function):
    """
    filters the graph according to edge weight and threshold_function
    :param graph: the aggregated knowledge graph
    :param threshold_function: function used to compute threshold
    :return:
    """
    relation_weights = graph.es['weight']
    threshold = threshold_function(relation_weights)
    logging.info('threshold ' + str(threshold))

    count_rels = len(graph.es)
    relations = graph.es
    filtered_graph = graph.copy()
    # remove all edges all edges - removing them on the fly causes strange bugs...
    filtered_graph.delete_edges(filtered_graph.es)
    assert not filtered_graph.es
    for relation in relations:
        if relation['weight'] >= threshold:
            filtered_graph.add_edge(relation.source, relation.target, **relation.attributes())
    filtered_count_rels = len(filtered_graph.es)
    logging.info(f'removed {count_rels - filtered_count_rels} relations')
    return filtered_graph


def __format_exp_to_dict_item(value, experiment_id):
    return {experiment_id: value}


def _format_attributes(attributes, experiment_id):
    """
    Adds a dict keyed by experiment_id for all attributes that can vary
    :param vertex:
    :param experiment_id:
    :return:
    """
    if attributes['type'] == 'qual_influence' or attributes['type'] == 'env_influence':
        attributes['value'] = __format_exp_to_dict_item(attributes['value'], experiment_id)
    elif attributes['type'] == 'parameter':
        attributes['user_value'] = __format_exp_to_dict_item(
            attributes['user_value'], experiment_id)
        attributes['original_value'] = __format_exp_to_dict_item(
            attributes['original_value'], experiment_id)
    else:
        raise ValueError('Unexpected type ' + attributes['type'])
    return attributes


def _merge_attributes(prev_attributes, new_attributes):
    """
    Accepts two dicts of attributes and merges them
    :param prev_attributes:
    :param new_attributes:
    :return:
    """
    merged_attributes = prev_attributes.copy()
    for key, value in new_attributes.items():
        # check if we have a dict we have to merge in both attribute dicts
        if isinstance(value, dict) and isinstance(prev_attributes[key],
                                                  dict):
            # merge the two dictionaries
            merged_attributes[key] = {**prev_attributes[key], **new_attributes[key]}
    return merged_attributes


def aggregate_unfiltered(graphs: Iterable[igraph.Graph], all_influences=False):
    """
    Aggregates many graphs to an all-encompassing knowledge graph
    :param graphs:
    :return:
    """
    from igraph import Graph
    knowledge_graph = Graph(directed=True)
    knowledge_graph['quality_aggregated'] = {}
    knowledge_graph['quality'] = {}
    knowledge_graph['environment_influences_lookup'] = {}
    knowledge_graph['quality_influences_lookup'] = {}
    knowledge_graph['all_parameters_lookup'] = {}
    knowledge_graph['environment_lookup'] = {}
    knowledge_graph['influenced_by_lookup'] = {}
    for graph in graphs:
        old_to_new_vertex_index = {}
        for vertex in graph.vs:
            formatted_attributes = _format_attributes(vertex.attributes(),
                                                      graph['experiment_id'])
            # check if there already is a vertex with the same name in kg
            try:
                vertex_in_kg = knowledge_graph.vs.find(vertex['key'])
            except ValueError as e:
                new_vertex = knowledge_graph.add_vertex(**formatted_attributes)
                old_to_new_vertex_index[vertex.index] = new_vertex.index
            else:
                vertex_in_kg.update_attributes(
                    _merge_attributes(vertex_in_kg.attributes(), formatted_attributes))
                old_to_new_vertex_index[vertex.index] = vertex_in_kg.index
        if not all_influences:
            for edge in graph.es:
                new_target = old_to_new_vertex_index[edge.target]
                new_source = old_to_new_vertex_index[edge.source]
                if edge.source_vertex['key'] != knowledge_graph.vs[new_source]['key'] or \
                        edge.target_vertex['key'] != knowledge_graph.vs[new_target]['key']:
                    raise Exception("source or target mapping is not consistent")
                try:
                    edge_id = knowledge_graph.get_eid(new_source, new_target)
                    knowledge_graph.es[edge_id]['weight'] += edge['weight']
                    knowledge_graph.es[edge_id]['experiments'].append(edge['experiment'])
                except Exception as e:
                    # edge doesn't exist -> create it
                    new_edge = knowledge_graph.add_edge(new_source, new_target)
                    new_edge['weight'] = edge['weight']
                    new_edge['experiments'] = [edge['experiment']]
        knowledge_graph['quality'].update({graph['experiment_id']: graph['quality']})
        knowledge_graph['quality_aggregated'].update({graph['experiment_id']: graph['quality_aggregated']})
        knowledge_graph['environment_influences_lookup'].update({graph['experiment_id']: graph['environment_influences']})
        knowledge_graph['quality_influences_lookup'].update({graph['experiment_id']: graph['quality_influences']})
        knowledge_graph['all_parameters_lookup'].update({graph['experiment_id']: graph['all_parameters']})
        knowledge_graph['environment_lookup'].update({graph['experiment_id']: graph['environment']})
        knowledge_graph['influenced_by_lookup'].update({graph['experiment_id']: graph['influenced_by']})
    if all_influences:
        parameters = knowledge_graph.vs.select(type='parameter')
        import igraph
        if type(parameters) is igraph.Vertex:
            parameters = [parameters]
        parameters_keys = [parameter for parameter in parameters]
        env_factors = knowledge_graph.vs.select(type='env_influence')
        if type(env_factors) is igraph.Vertex:
            env_factors = [env_factors]
        qual_factors = knowledge_graph.vs.select(type='qual_influence')
        if type(qual_factors) is igraph.Vertex:
            qual_factors = [qual_factors]
        influences_keys = [e for e in env_factors] + [q for q in qual_factors]
        edges = itertools.product(influences_keys, parameters_keys)
        knowledge_graph.add_edges(edges)
    if not all_influences:
        # normalize weights by amounts of subgraphs #TODO probably problematic if there are only a few subgraphs with high confidence for certain areas
        knowledge_graph.es['weight'] = list(map(lambda x: x / len(graphs), knowledge_graph.es['weight']))
    return knowledge_graph


def _get_value(vertex, experiment_id, verbose=False):
    """
    retrieves the value of the given vertex for the given experiment_id
    :param vertex:
    :param experiment_id:
    :return: Tupel (user_value, original_value) for parameters, (value, none) for influences
    """
    if vertex['type'] == 'parameter':
        try:
            return float(vertex['user_value'][experiment_id]), float(vertex['original_value'][experiment_id])
        except KeyError:
            # this influence hasn't been highlighted for this experiment
            return None, None
        except ValueError:
            return vertex['user_value'][experiment_id], vertex['original_value'][experiment_id]
    elif vertex['type'] == 'qual_influence' or vertex['type'] == 'env_influence':
        try:
            return float(vertex['value'][experiment_id]), None
        except KeyError:
            # this influence hasn't been highlighted for this experiment
            return None, None
        except ValueError as e:
            if verbose:
                logging.warning("non floatable value " + vertex['user_value'][experiment_id] + ' encountered. ' + e)
            return vertex['user_value'][experiment_id], vertex['original_value'][experiment_id]
    else:
        raise Exception('Type unknown: ' + vertex['type'])


def _get_experiments_in_vertex(vertex):
    if vertex['type'] == 'parameter':
        return vertex['user_value'].keys
    elif vertex['type'] == 'qual_influence' or vertex['type'] == 'env_influence':
        return vertex['value'].keys
    else:
        raise Exception('Type unknown: ' + vertex['type'])


@deprecation.deprecated()
def prepare_data_for_filtering_calculations(graph, experiment_series, fill_nones=True, include_changed_parameters=False,
                                            include_influences=True):
    """
    TODO maybe we should't reconstruct this data out of the graph but rather is it as provided by preprocessing or similar
    only works for numerical values!
    :param graph:
    :param experiment_series:
    :param fill_nones:
    :param relative:
    :return:
    """
    e_ps = defaultdict(dict)
    # prepare a dict mapping experiment series ids to a dict of their experiments with respective ratings - required to filter for best only
    # all parameters contained in the graph as vertices containing user_value dictionary in format exp_id: value
    parameters = [vertex for vertex in graph.vs if vertex['type'] == 'parameter']
    influences = [vertex for vertex in graph.vs if
                  vertex['type'] == 'env_influence' or vertex['type'] == 'qual_influence']

    para_n_inf_per_series = _calc_params_and_infs_per_series_simple(experiment_series,
                                                                    influences, parameters)

    x_dicts = []
    y_dicts = []
    index = []
    for series_id, exp_ids in experiment_series.items():
        for exp_id in exp_ids:
            y_dicts.append({k: v for k, v in graph['quality'][exp_id].items()})
            index.append(exp_id)
            exp_dict = {}
            if include_changed_parameters:
                for parameter in parameters:
                    param_value_user, param_value_original = _get_value(parameter, exp_id)
                    if fill_nones:
                        param_value_user = _fill_nones_param(exp_id, graph, para_n_inf_per_series, param_value_user,
                                                             parameter, series_id)
                    if param_value_user is not None and param_value_original is not None:
                        exp_dict.update({parameter['key'] + '_adjusted_value': param_value_user,
                                         parameter['key'] + '_original_value': param_value_original,
                                         parameter['key'] + '_relative': param_value_user - param_value_original})
            if include_influences:
                for influence in influences:
                    influence_value, _ = _get_value(influence, exp_id)
                    if fill_nones:
                        influence_value = _fill_nones_influence(exp_id, graph, influence, influence_value,
                                                                para_n_inf_per_series, series_id)
                    if influence_value is not None:
                        exp_dict.update({influence['key'] + '_influence': influence_value})
                        if influence['type'] == 'qual_influence':
                            # check if the key is present in the quality rating of the previous experiment
                            if influence['key'] in graph['quality'][exp_id]:
                                exp_dict.update({influence['key'] + '_relative': -1 * (
                                        graph['quality'][exp_id][influence['key']] - influence_value)})
                            else:
                                raise ValueError(influence[
                                                     'key'] + ' of experiment ' + exp_id + ' not present in quality characteristics of referenced experiment')
            x_dicts.append(exp_dict)
    x = pandas.DataFrame.from_records(x_dicts, index=index)
    y = pandas.DataFrame.from_records(y_dicts, index=index)
    return x, y


def prepare_dfs_of_experiments(graph: igraph.Graph, experiment_series, fill_nones=False,
                               compute_changed_parameters_relative_to_defaults=False, encode_strings=False):
    """
    constructs dataframes with changed parameters and influences as present in influences section of experiments
    Note: it is not guaranteed that relative parameters are present for all parameters (occurs if there is no preceeding experiment)!
    :param graph:
    :param experiment_series:
    :param fill_nones: should be false if we want to highlight the users' input. otherwise it also contains values that are not selected as influences
    :param compute_changed_parameters_relative_to_defaults: calculates the changed parameters to the default parametrization. Otherwise they are calculated to the preceeding experiment resulting in a much sparser but more expressive df (influences etc are also calculated in regards to the preceeding exp)
    :return:
    """
    exp_to_preceeding_exp = __calculate_preceeding_experiments_lookup(experiment_series, graph)

    # prepare a dict mapping experiment series ids to a dict of their experiments with respective ratings - required to filter for best only
    # all parameters contained in the graph as vertices containing user_value dictionary in format exp_id: value
    parameters = [vertex for vertex in graph.vs if vertex['type'] == 'parameter']
    influences = [vertex for vertex in graph.vs if
                  vertex['type'] == 'env_influence' or vertex['type'] == 'qual_influence']

    para_n_inf_per_series = _calc_params_and_infs_per_series_simple(experiment_series,
                                                                    influences, parameters)

    index = defaultdict(list)
    # lists to collect data of which dfs are created
    quality = []
    quality_relative = []
    changed_parameters = []
    changed_parameters_relative = []
    influences_ = []
    influences_relative = []
    # TODO refactor, these should be parameters
    datacollection = {'quality': quality, 'quality_relative': quality_relative,
                      'changed_parameters': changed_parameters,
                      'changed_parameters_relative': changed_parameters_relative, 'influences': influences_,
                      'influences_relative': influences_relative}
    for series_id, exp_ids in experiment_series.items():
        for exp_id in exp_ids:
            # set up indices
            for key in datacollection.keys():
                index[key].append(exp_id)
            exp_quality = {}
            exp_quality_relative = {}
            exp_changed_parameters = {}
            exp_changed_parameters_relative = {}
            exp_influences = {}
            exp_influences_relative = {}
            exp_datacollection = {'quality': exp_quality,
                                  'quality_relative': exp_quality_relative,
                                  'changed_parameters': exp_changed_parameters,
                                  'changed_parameters_relative': exp_changed_parameters_relative,
                                  'influences': exp_influences,
                                  'influences_relative': exp_influences_relative}
            for quality_attribute, rating in graph['quality'][exp_id].items():
                exp_quality.update({quality_attribute: rating})
                # quality relative:
                try:
                    relative_rating = -1 * (graph['quality'][exp_id][quality_attribute] -
                                            graph['quality'][exp_to_preceeding_exp[exp_id]][quality_attribute])
                except KeyError:
                    relative_rating = None
                exp_quality_relative.update({quality_attribute: relative_rating})

            # changed parameters
            for parameter in parameters:
                param_value_user, param_value_original = _get_value(parameter, exp_id)
                if fill_nones:
                    param_value_user = _fill_nones_param(exp_id, graph, para_n_inf_per_series, param_value_user,
                                                         parameter, series_id)
                if isinstance(param_value_user, str):
                    raise ValueError(
                        'string encountered where it shouldnt have occured - replace Exception with label_encoding')
                else:
                    user_value = param_value_user
                exp_changed_parameters.update({parameter['key']: user_value})
                if compute_changed_parameters_relative_to_defaults:
                    # calculate changed params relative based on default values
                    if param_value_user is not None and param_value_original is not None:
                        exp_changed_parameters.update({parameter['key']: param_value_user})
                        if param_value_original is not None:
                            exp_changed_parameters_relative.update(
                                {parameter['key']: param_value_user - param_value_original})
                else:
                    if param_value_user is not None:
                        try:
                            previous_value = graph['all_parameters_lookup'][exp_to_preceeding_exp[exp_id]][
                                parameter['key']]
                        except KeyError:
                            # silent fail if there is no preceeding experiment
                            continue
                        if isinstance(previous_value, bool):
                            # if previous value is bool uservalue also has to be bool
                            if isinstance(param_value_user, str):
                                user_value = bool(util.strtobool(param_value_user))
                            else:
                                user_value = bool(param_value_user)
                            # calculate change of bool parameter - the behaviour is ok since we currently only focus on non-relative values during rule creation (refer to relation_to_rule)
                            exp_changed_parameters_relative.update(
                                {parameter['key']: float(user_value) - float(previous_value)})
                        elif isinstance(previous_value, str):
                            # TODO XAI-632: when this will be fixed, use AipeDataProvider.get_label_encoder() to avoid passing label encoder everywhere again
                            # TODO should we encode all_parameters_lookup as well during graph creation to get around passing label_encoder to absolutely everything?
                            # we encountered an unfloatable value -> use label_encoder to determine label encoded value
                            # previous_value = label_encoder.transform(previous_value.lower().replace(" ", ""), parameter['key'])
                            # TODO XAI-587: subtracting unordered values makes no sense -> keep the user value
                            # TODO XAI-558 should we keep this behaviour or simply ignore encoded values for
                            exp_changed_parameters_relative.update(
                                {parameter['key']: user_value})
                            # exp_changed_parameters_relative.update(
                            #    {parameter['key']: previous_value - user_value})
                        else:
                            exp_changed_parameters_relative.update(
                                {parameter['key']: param_value_user - previous_value})


            # influences:
            for influence in influences:
                influence_value, _ = _get_value(influence, exp_id)
                if fill_nones:
                    influence_value = _fill_nones_influence(exp_id, graph, influence, influence_value,
                                                            para_n_inf_per_series, series_id)
                if influence_value is not None:
                    exp_influences.update({influence['key']: influence_value})
                    if influence['type'] == 'qual_influence':
                        # check if the key is present in the quality rating of the previous experiment
                        # TODO decide whether this is nonsense or not - the information is duplicated by quality_relative but it highlights the influences...
                        if influence['key'] in graph['quality'][exp_id]:
                            exp_influences_relative.update({influence['key']: -1 * (
                                    graph['quality'][exp_id][influence['key']] - influence_value)})
                        else:
                            raise ValueError(influence[
                                                 'key'] + ' of experiment ' + exp_id + ' not present in quality characteristics of referenced experiment')
            for data_key, collection in datacollection.items():
                collection.append(exp_datacollection[data_key])
    dfs = defaultdict(pandas.DataFrame)
    for key, data in datacollection.items():
        # convert to dataframe and replace Nones with NaNs
        dfs[key] = pandas.DataFrame.from_records(data, index=index[key]).fillna(value=np.nan)
    return dfs


def __calculate_preceeding_experiments_lookup(experiment_series, graph):
    # determine the preceeding experiment in a series to allow relative quality calculations
    experiment_to_experiment_series = {exp: exp_ser for exp_ser, exps in experiment_series.items() for exp in exps}
    exp_to_preceeding_exp = defaultdict(int)
    for exp, series in experiment_to_experiment_series.items():
        # first we have a look if the experiment in question has a link to a preceeding experiment contained in the influences
        preceeding_exp = graph['influenced_by_lookup'][exp]
        if preceeding_exp is not None:
            exp_to_preceeding_exp.update({exp: int(preceeding_exp)})
        else:
            # if that is not the case we try to determine it based on ids in the experiment series
            experiments = np.sort(experiment_series[series])  # just to be on the safe side..
            experiments = list(experiments)
            index_of_exp = experiments.index(exp)
            if (index_of_exp > 0):
                preceeding_exp = experiments[index_of_exp - 1]
                exp_to_preceeding_exp.update({exp: int(preceeding_exp)})
    exp_to_preceeding_exp = dict(exp_to_preceeding_exp)
    return exp_to_preceeding_exp


def fill_nans_of_clustering_df(graph, df, type, active_fill=False):
    """
    tries to fill the very sparse data present in influences
    #TODO maybe it would be better to prepare different dataframes
    :param active_fill: actively searching the lookup tables for values. This results in values of parameters/influences being added that are present but not selected by the user!
    :param default_fill: what to fill nans/Nones with
    :param graph: graph containing lookups for experiments
    :param df:
    :param type: which data to except in dataframe. E.g. influences, changed_parameters etc
    :return: df with filled nans, df only containing dense regions (=parameters & influences)
    """
    filled_df = df.copy()
    # TODO should we consider the total change in quality? -> we would need to pass the fully quality information of the previous experiment
    # TODO probably we should - do we already get that by "include all influences=true?!" - Keep temperature in mind - it should also be returned by include all influences!
    if type == 'changed_parameters':
        for parameter in df:
            for experiment, value in df[parameter].items():
                # TODO refactor to use fillnones_influences/parameters - why is experiment series etc needed?!
                try:
                    nanvalue = np.isnan(value)
                    if active_fill and np.isnan(value):
                        try:
                            value = float(graph['all_parameters_lookup'][experiment][parameter])
                            filled_df[parameter][experiment] = value
                        except KeyError:
                            continue
                        except ValueError:
                            # raised if parameter is categorical/string based
                            continue
                except Exception as e:
                    # raised if parameter is categorical/string based
                    logging.warning(e)
    elif type == 'influences':
        for influence in df:
            for experiment, value in df[influence].items():
                # TODO refactor to use fillnones_influences/parameters - why is experiment series etc needed?!
                if active_fill and np.isnan(value):
                    try:
                        value = float(
                            graph['environment_influences_lookup'][experiment][influence]['value'])
                        filled_df[influence][experiment] = value
                    except KeyError:
                        continue
                    except ValueError:
                        # raised if parameter is non categorical/string based
                        continue
        env_factors = ['temperature', 'humidity']
        for env_factor in env_factors:
            # in try except bc temperature or humidity might not be present
            try:
                env_facor_series = df[env_factor]
                for experiment, value in env_facor_series.items():
                    if active_fill and np.isnan(value):
                        try:
                            value = float(graph['environment_lookup'][experiment][env_factor])
                            filled_df[env_factor][experiment] = value
                        except KeyError:
                            continue
                        except ValueError:
                            # raised if parameter is non categorical/string based
                            continue
            except KeyError:
                continue
    elif type == 'changed_parameters_relative':
        for parameter in df:
            for experiment, value in df[parameter].items():
                if active_fill and np.isnan(value):
                    value = 0
                    filled_df[parameter][experiment] = value
    else:
        pass
    return filled_df


@deprecation.deprecated(details="use prepare_dfs_of_experiments instead")
def extract_e_p_values(graph, experiment_series, best_only=False, fill_nones=False):
    """
    Extracts a dictionary with dataframes with concrete values of the parameter-influence pairs of a series
    :param graph:
    :param experiment_series:
    :param best_only: only include the best/last iteration/experiment of an experiment series
    :return:
    """
    # prepare a dict mapping experiment series ids to a dict of their experiments with respective ratings - required to filter for best only
    experiment_series_with_ratings = {
        experiment_serie: {experiment_id: graph['quality_aggregated'][experiment_id] for experiment_id in
                           experiment_ids} for
        experiment_serie, experiment_ids in experiment_series.items()}

    # all parameters contained in the graph as vertices containing user_value dictionary in format exp_id: value
    parameters = [vertex for vertex in graph.vs if vertex['type'] == 'parameter']
    influences_for_parameters = {parameter.index: [edge.source_vertex for edge in parameter.in_edges()] for parameter in
                                 parameters}

    es_ps_data = defaultdict(list)
    # we need a way to check if a parameter is present in the series - otherwise we add a lot of series that do not have anythin in common with what we want
    para_n_inf_per_series = _calc_params_and_infs_per_series(experiment_series, influences_for_parameters,
                                                             parameters)

    if best_only:
        expid_with_min_rating_for_series = {
            series_id: min(experiments_ratings.items(), key=operator.itemgetter(1))[0] for
            series_id, experiments_ratings in experiment_series_with_ratings.items()}
        for series_id, exp_id in expid_with_min_rating_for_series.items():
            for parameter in parameters:
                for influence in influences_for_parameters[parameter.index]:
                    param_value_user, param_value_original = _get_value(parameter, exp_id)
                    influence_value, _ = _get_value(influence, exp_id)
                    if fill_nones:
                        influence_value = _fill_nones_influence(exp_id, graph, influence, influence_value,
                                                                para_n_inf_per_series, series_id)
                        param_value_user = _fill_nones_param(exp_id, graph, para_n_inf_per_series, param_value_user,
                                                             parameter, series_id)

                    exp_dict = {'series_id': series_id, 'exp_id': exp_id,
                                'param_value': param_value_user,
                                'param_value_original': param_value_original,
                                'influence_value': influence_value,
                                'quality_aggregated': graph['quality_aggregated'][exp_id]}
                    quality_dict = {k: v for k, v in graph["quality"][exp_id].items()}
                    exp_dict.update(quality_dict)
                    es_ps_data[parameter['name'] + '-' + influence['name']].append(exp_dict)
    else:
        for series_id, exp_ids in experiment_series.items():
            for exp_id in exp_ids:
                for parameter in parameters:
                    for influence in influences_for_parameters[parameter.index]:
                        param_value_user, param_value_original = _get_value(parameter, exp_id)
                        influence_value, _ = _get_value(influence, exp_id)
                        # revert to all recorded influences/parameters to fill nones
                        if fill_nones:
                            influence_value = _fill_nones_influence(exp_id, graph, influence, influence_value,
                                                                    para_n_inf_per_series, series_id)
                            param_value_user = _fill_nones_param(exp_id, graph, para_n_inf_per_series, param_value_user,
                                                                 parameter, series_id)

                            exp_dict = {'series_id': series_id, 'exp_id': exp_id,
                                        'param_value': param_value_user,
                                        'param_value_original': param_value_original,
                                        'influence_value': influence_value,
                                        'quality_aggregated': graph['quality_aggregated'][exp_id]}

                        else:
                            exp_dict = {'series_id': series_id, 'exp_id': exp_id,
                                        'param_value': param_value_user,
                                        'param_value_original': param_value_original,
                                        'influence_value': influence_value,
                                        'quality_aggregated': graph['quality_aggregated'][exp_id]}
                        quality_dict = {k: v for k, v in graph["quality"][exp_id].items()}
                        exp_dict.update(quality_dict)
                        es_ps_data[parameter['name'] + '-' + influence['name']].append(exp_dict)
    # TODO check if we're discarding too much. Maybe it's better to delete columns altough exp series should always be created with the same cura version and therefore not exhibit nones
    es_ps_df = {e_p: pandas.DataFrame(data).dropna() for e_p, data in es_ps_data.items()}
    return es_ps_df


def _calc_params_and_infs_per_series(experiment_series, influences_for_parameters, parameters):
    """
    Calculates a dict mapping series id to set of changed parameters and highlighted influences
    TODO is this unneccessarily complex and could be calculated much simpler by iterating through all parameters and all influences?
    :param experiment_series:
    :param influences_for_parameters:
    :param parameters:
    :return:
    """
    para_n_inf_per_series = defaultdict(set)
    for series_id, exp_ids in experiment_series.items():
        for exp_id in exp_ids:
            for parameter in parameters:
                param_value_user, _ = _get_value(parameter, exp_id)
                if not param_value_user is None:
                    para_n_inf_per_series[series_id].add(parameter['key'])
                for influence in influences_for_parameters[parameter.index]:
                    influence_value, _ = _get_value(influence, exp_id)
                    if not influence_value is None:
                        para_n_inf_per_series[series_id].add(influence['key'])
    return para_n_inf_per_series


def _calc_params_and_infs_per_series_simple(experiment_series, influences, parameters):
    """
    Calculates a dict mapping series id to set of changed parameters and highlighted influences
    TODO is this unneccessarily complex and could be calculated much simpler by iterating through all parameters and all influences?
    :param experiment_series:
    :param influences_for_parameters:
    :param parameters:
    :return:
    """
    para_n_inf_per_series = defaultdict(set)
    for series_id, exp_ids in experiment_series.items():
        for exp_id in exp_ids:
            for parameter in parameters:
                param_value_user, _ = _get_value(parameter, exp_id)
                if not param_value_user is None:
                    para_n_inf_per_series[series_id].add(parameter['key'])
                for influence in influences:
                    influence_value, _ = _get_value(influence, exp_id)
                    if not influence_value is None:
                        para_n_inf_per_series[series_id].add(influence['key'])
    return para_n_inf_per_series


def _fill_nones_param(exp_id, graph, para_n_inf_per_series, param_value_user, parameter_vertex,
                      series_id):
    """
    Try to fill as many Nones as possible by consulting lookup tables for all parameters / influences
    only works for user param values
    :param exp_id:
    :param graph:
    :param influence:
    :param influence_value:
    :param para_n_inf_per_series:
    :param param_value_user:
    :param parameter_vertex:
    :param series_id:
    :return:
    """
    if param_value_user is None and parameter_vertex['key'] in para_n_inf_per_series[series_id]:
        # we still get some Nones since we can't generate the parameters if they do not exist in all
        # _parameters. This could happen if a different cura version or printer is used.
        if parameter_vertex['key'] not in graph['all_parameters_lookup'][exp_id] or parameter_vertex['key'] is None:
            return None
        else:
            try:
                param_value_user = float(graph['all_parameters_lookup'][exp_id][parameter_vertex['key']])
            except KeyError:
                return None
            except ValueError:
                return None
    return param_value_user


def _fill_nones_influence(exp_id, graph, influence, influence_value, para_n_inf_per_series,
                          series_id):
    """
    Try to fill as many Nones as possible by consulting lookup tables for all parameters / influences
    only works for user param values
    :param exp_id:
    :param graph:
    :param influence:
    :param influence_value:
    :param para_n_inf_per_series:
    :param param_value_user:
    :param parameter:
    :param series_id:
    :return:
    """

    if influence_value is None and influence['key'] in para_n_inf_per_series[series_id]:
        # we still get some Nones since we can't generate the influence's values out of thin air.
        # However, since this most likely affects different configurations it is unlikely that we have
        # differing influences inside an experiment series.
        if influence['key'] not in graph['environment_influences_lookup'][exp_id]:
            return None
        else:
            # TODO reformat environment_influences_lookup i.e. flatten disregard influential values
            try:
                influence_value = float(graph['environment_influences_lookup'][exp_id][influence['key']]['value'])
            except KeyError:
                return None
    return influence_value


def filter_knowledge_graph_for_influence(graph, influence_key):
    influence = graph.vs.find(key=influence_key)
    neighbors = graph.neighborhood(vertices=influence, order=1, mode="ALL", mindist=0)
    return graph.subgraph(neighbors)


def update_experiment_series(graph_list: Iterable[igraph.Graph], experiment_series:dict =None):
    """
    update_experiment_series(graph_list, experiment_series = dict())
    Is needed if there are experiment id's in the experiment_series dictionary experiment that are not in the graph list
    because these experiments were filtered out
    :param graph_list list of all graphs that will be used to aggregate a complete knowledge graph
    :param experiment_series dictionary that contains the information about all experiment series with the id's
    :return returns the updated experiment series dict
    """
    if experiment_series is None:
        experiment_series = dict()
    for series in list(experiment_series):
        for experiment in reversed(experiment_series[series]):
            is_in_graph = False
            for graph in graph_list:
                if experiment == graph['experiment_id']:
                    is_in_graph = True
                    break
            if is_in_graph is False:
                experiment_series[series].remove(experiment)
            if len(experiment_series[series]) == 0:
                experiment_series.pop(series)
    return experiment_series