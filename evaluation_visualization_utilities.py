from __future__ import annotations

import logging
import os
import numpy as np
import igraph
import networkx as nx
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import graph_visualisation
import knowledge_graph as kg
from rule_base.rule import Rule
from rule_base.rule_extraction import calculate_bins, rule_from_edge
from utilities import Path


def visualize_all_graphs(original_graph: igraph.Graph,
                         filtered_graphs: igraph.Graph,
                         experiment_series: dict = None):
    """
    creates html files for the knowledge graphs and stores them in a sorted directory structure
    Given graphs are supposed to be in igraph format, they will be converted to networkx graphs inside this function
    :param original_graph base knowledge graph
    :param filtered_graphs list of filtered knowledge graph
    """
    # create networkx graph from base knowledge graph in igraph format
    if experiment_series is None:
        experiment_series = dict()
    nx_original_graph = nx.from_numpy_array(
        np.array(original_graph.get_adjacency().data))
    node_attributes = get_node_attributes_for_graph_visualization(
        original_graph)
    edge_attributes = get_edge_attributes_for_graph_visualization(
        original_graph, experiment_series)
    nx.set_edge_attributes(nx_original_graph, edge_attributes)
    nx.set_node_attributes(nx_original_graph, node_attributes)

    # remember the positions of the nodes to plot them also in the filtered graphs
    positions = graph_visualisation.plot_original_graph(nx_original_graph)
    for weight_method, filter_method, graph, filtered_series in filtered_graphs:
        filename = f'figs{os.sep}graphs{os.sep}{weight_method}{os.sep}{filter_method}' + "_graph.html"
        visualize_graph_filtering(original_graph,
                                  graph,
                                  positions,
                                  experiment_series,
                                  fn=filename)


def visualize_graph_filtering(params_dict: dict, positions=None, fn=''):
    """
    calculate a diff graph from a base kg graph and a filtered kg graph
    If no positions are given both kg will be plotted otherwise only the filtered graph with the given positions
    kgs are given as igraph
    :param graph_before_filter base kg
    :param graph_after_filter filtered kg
    :param positions positions of the vertices in the base graph
    :param experiment_series experiment series to get the relative changed parameters
    :param fn filename for the html files to be stored. If not given the will just be stored in the current directory
    """
    # create networkx graphs from igraph
    if positions is None:
        positions = []
    np_before_filter = np.array(params_dict['base_graph'].get_adjacency().data)
    np_after_filter = np.array(
        params_dict['filtered_graph'].get_adjacency().data)

    nx_before_filter = nx.from_numpy_array(np_before_filter)
    nx_after_filter = nx.from_numpy_array(np_after_filter)

    # get attributes for the nodes and the edges (colors, hover information, etc.)
    edge_attributes = get_edge_attributes_for_graph_visualization(
        params_dict['base_graph'], params_dict['exp_series'])
    node_attributes = get_node_attributes_for_graph_visualization(
        params_dict['base_graph'])

    # calculate graph diff
    graph_diff = graph_visualisation.diff_graphs(nx_before_filter,
                                                 nx_after_filter)

    # set the attributes for the edges and nodes
    nx.set_node_attributes(nx_before_filter, node_attributes)
    nx.set_node_attributes(graph_diff, node_attributes)
    nx.set_edge_attributes(nx_before_filter, edge_attributes)
    nx.set_edge_attributes(graph_diff, edge_attributes)
    if len(positions) == 0:
        graph_visualisation.plot_graphs_same_positions(
            [nx_before_filter, graph_diff], highlight_filtered_edges=True)
    else:
        graph_visualisation.plot_graph(graph_diff,
                                       fn,
                                       positions,
                                       highlight_filtered_edges=True)


def get_node_attributes_for_graph_visualization(graph: igraph.Graph):
    """
    get node attributes for graph visualisation
    These can be used for hover information, colors etc.
    """
    attributes = dict()
    for idx, vertex in enumerate(graph.vs):
        attributes[idx] = dict()
        if vertex['type'] == 'parameter':
            attributes[idx]['node_color'] = 'blue'
        elif vertex['type'] == 'qual_influence':
            attributes[idx]['node_color'] = 'yellow'
        elif vertex['type'] == 'env_influence':
            attributes[idx]['node_color'] = 'red'
        attributes[idx]['hover_name'] = vertex['name']
        attributes[idx]['type'] = vertex['type']
    return attributes


def get_edge_attributes_for_graph_visualization(graph: igraph.Graph,
                                                experiment_series,
                                                verbose=False):
    """
    get edge attributes for graph visualisation
    These can be used for hover information, colors etc.
    """
    all_dfs = kg.prepare_dfs_of_experiments(graph,
                                            experiment_series,
                                            fill_nones=False)
    relative_params = all_dfs['changed_parameters_relative']
    params = all_dfs['changed_parameters']
    influences = all_dfs['influences']
    influence_bin_data = calculate_bins(all_dfs)

    attributes = dict()
    for edge in graph.es:
        key = (edge.source, edge.target)
        attributes[key] = dict()
        influence = edge.source_vertex['name']
        parameter = edge.target_vertex['name']
        attributes[key]['source_vertex'] = influence
        attributes[key]['target_vertex'] = parameter
        try:
            rule_str = str(
                rule_from_edge(relative_params, params, influences,
                               influence_bin_data, influence, parameter,
                               Rule.ParamType.UNKNOWN))
            attributes[key]['rule'] = rule_str
        except Exception as e:
            if verbose:
                logging.warning(
                    f'No rule for the relation {influence} - {parameter}. {e}')
    return attributes


def visualize_performance_of_metric(fn: Path,
                                    data: dict,
                                    with_curve_fit=False,
                                    metric='',
                                    plot_baseline=False):
    """
    visualizes the performance of a certain metric
    :param fn filename with path where to store the generated figure
    :param data dictionary that contains a 'x' and a 'y' key which then contain an iterable with all the values to be plotted
    :param with_curve_fit if true then a log-based function will be fitted in the given points otherwise only a scatter plot will be created
    :param metric name of the metric that is used for labelling the y axis of the figure
    :param plot_baseline true if baseline should be plotted to every set of data
    """
    fig, ax = plt.subplots()
    ax = create_metric_performance_plot(data,
                                        with_curve_fit,
                                        metric,
                                        plot_baseline,
                                        ax=ax)
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    fig.savefig(fn, bbox_inches='tight')
    plt.show()


def create_metric_performance_plot(data: dict,
                                   with_curve_fit=False,
                                   metric='',
                                   plot_baseline=False,
                                   ax=None):
    ax = ax or plt.gca()

    def func(x, a, b):
        return a * np.log(x) + b

    markers = ['+', 'x', '.']
    colours = ['b', 'g', 'r']
    line_type = ['-', '--', '-.']
    for idx, level in enumerate(data['metric_data']):
        x = np.array(data['metric_data'][level]['x'])
        y = np.array(data['metric_data'][level]['y'])
        x_baseline = np.array(data['baseline'][level]['x'])
        y_baseline = np.array(data['baseline'][level]['y'])
        if with_curve_fit is True:
            popt, pcov = curve_fit(func, x, y)
            if plot_baseline is True:
                popt_base, pcov_base = curve_fit(func, x_baseline, y_baseline)
        label = level.lower()
        if with_curve_fit is False:
            ax.scatter(x, y, label=label, marker=markers[idx], c=colours[idx])
            if plot_baseline is True:
                ax.scatter(x_baseline,
                           y_baseline,
                           label='baseline\n' + label,
                           marker=markers[idx],
                           c='k')
        else:
            ax.plot(x, y, '+', marker=markers[idx], c=colours[idx])
            ax.plot(x,
                    func(x, *popt),
                    label=label,
                    c=colours[idx],
                    ls=line_type[idx])
            if plot_baseline is True:
                ax.plot(x_baseline,
                        y_baseline,
                        '+',
                        marker=markers[idx],
                        c='k')
                ax.plot(x_baseline,
                        func(x_baseline, *popt_base),
                        label='baseline\n' + label,
                        c='k',
                        ls=line_type[idx])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    level = 'high'
    ax.set_xlim([
        min(data['metric_data'][level]['x']),
        max(data['metric_data'][level]['x']) + 10
    ])
    ax.set_xlabel('Number of considered experiments')
    ax.set_ylabel(metric.split('/')[1])
    return ax
