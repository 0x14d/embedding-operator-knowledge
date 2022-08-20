from pprint import pprint

import networkx as nx
import pandas

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool, LabelSet, ColumnDataSource)
from bokeh.plotting import from_networkx
import numpy as np
import os


def plot_graph(graph, fname, positions, highlight_filtered_edges=False):
    """
    Plots a networkx graph
    :param graph: networkx graph to plot
    :param fname: filename to save the plot to
    :param positions: node-position dict
    :param highlight_filtered_edges: whether to highlight filtered edges. Requires attriute 'filtered' on all edges.
    :return:
    """
    plot = Plot(plot_width=1400, plot_height=1400,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    plot.title.text = "Graph Interaction Demonstration"

    # graph_renderer doesn't auto update -> we have to repeat the condition or find a way to call an update method
    # check in dictionary (position 2 of the tuple) of the first edge if we have a filtered graph
    if highlight_filtered_edges and 'filtered' in list(graph.edges.data())[0][2]:
        BLACK, RED = "black", "red"
        SOLID_LINE, DASHED_LINE = "solid", "dashed"
        edge_colors = {}
        edge_dashs = {}
        for start_node, end_node, attributes in graph.edges(data=True):
            edge_color = BLACK if attributes['filtered'] is False else RED
            edge_dash = SOLID_LINE if attributes['filtered'] is False else DASHED_LINE
            edge_colors[(start_node, end_node)] = edge_color
            edge_dashs[(start_node, end_node)] = edge_dash

        nx.set_edge_attributes(graph, edge_colors, 'edge_color')
        nx.set_edge_attributes(graph, edge_dashs, 'edge_dash')
    graph_renderer = from_networkx(graph, positions, scale=0.5, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color='node_color')
    # if we want to highlight edges and there is an attribute 'filtered' highlight them
    if highlight_filtered_edges and 'filtered' in list(graph.edges.data())[0][2]:
        graph_renderer.edge_renderer.glyph = MultiLine(line_color='edge_color', line_alpha=0.8, line_width="weight", line_dash = 'edge_dash')
    else:
        graph_renderer.edge_renderer.glyph = MultiLine(line_alpha=0.8, line_width="weight")

    # node labels:
    x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
    node_labels = list(graph.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(node_labels))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px',
                      background_fill_alpha=.7)

    node_hover_tool = HoverTool(tooltips=[("name", "@hover_name"), ("type", "@type")], renderers=[graph_renderer.node_renderer])
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
    edge_hover_tool = HoverTool(tooltips=[("source", "@source_vertex"), ("target", "@target_vertex"), ("rule", "@rule")],
                                renderers=[graph_renderer.edge_renderer], line_policy='interp')
    plot.add_tools(edge_hover_tool)

    plot.renderers.append(labels)
    plot.renderers.append(graph_renderer)
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    output_file(fname)
    show(plot)


def plot_graphs_same_positions(graphs, path='', highlight_filtered_edges=False):
    """
    Plots multiple graphs and makes sure that the nodes are on the same positions. Useful when we want to visualise changes to a graph.
    :param graphs: list of networkx graphs
    :param path: path to where the graph should be saved
    :param highlight_filtered_edges: whether to highlight edges that are present in one but not the other
    :return:
    """
    positions = None
    if len(graphs) > 1:
        first = graphs[0]
        positions = nx.spring_layout(first)
        pprint(positions)
    if positions is not None:
        for i, g in enumerate(graphs):
            plot_graph(g, fname=f'./{i}.html', positions=positions, highlight_filtered_edges=highlight_filtered_edges)


def plot_original_graph(graph):
    """
    plot the original graph and returns the positions of the nodes
    """
    positions = nx.spring_layout(graph)
    plot_graph(graph, 'figs/graphs/original_graph.html', positions)
    return positions


def diff_graphs(unfiltered_graph, filtered_graph):
    """
    Computes a graph that is suited to highlight which edges where removed during filtering.
    :param unfiltered_graph: networkx graph
    :param filtered_graph: networkx graph
    :return: diff_graph: a graph based on unfiltered_graph with an attribute 'filtered' added to edges. It is True if they are present in the unfiltered_graph but not in filtered_graph.
    """
    diff_graph = unfiltered_graph.copy()
    unfiltered_graph_edges = set(unfiltered_graph.edges)
    filtered_graph_edges = set(filtered_graph.edges)
    filtered_edges = unfiltered_graph_edges - filtered_graph_edges

    nx.set_edge_attributes(diff_graph,
                           {edge: True if edge in filtered_edges else False for edge in list(diff_graph.edges)},
                           'filtered')

    return diff_graph


if __name__ == '__main__':
    graph1 = np.array([
        [1, 2, 3],
        [2, 1, 4],
        [3, 4, 1]
    ])
    filtered_g1 = np.array([
        [1, 0, 3],
        [0, 1, 4],
        [3, 4, 1]
    ])
    graph2 = np.array([
        [1, 5, 6],
        [5, 1, 8],
        [6, 8, 1]
    ])

    graphlabels = ['node1', 'node2', 'node3']

    graph1_graph = nx.from_pandas_adjacency(pandas.DataFrame(graph1, index=graphlabels, columns=graphlabels))
    filtered_g1_graph = nx.from_pandas_adjacency(pandas.DataFrame(filtered_g1, index=graphlabels, columns=graphlabels))
    graph2_graph = nx.from_pandas_adjacency(pandas.DataFrame(graph2, index=graphlabels, columns=graphlabels))

    diff_graph = diff_graphs(graph1_graph, filtered_g1_graph)
    plot_graphs_same_positions([graph1_graph, filtered_g1_graph, diff_graph], highlight_filtered_edges=True)
