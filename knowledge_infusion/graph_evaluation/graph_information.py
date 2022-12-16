"""
This module pickles meta-information about graphs
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
import pickle

def generate_graph_information(
    knowledge_graph_type,
    edges,
    metadata,
    train_data,
    test_data,
    base_folder
) -> None:
    """Generates graph information and stores it in a pickle

    Args:
        knowledge_graph_type (str): The representation 
        edges (pandas.Dataframe): The edges present in the graph
        metadata (pandas.Dataframe): The nodes present in the graph
        train_data (pykeen.TriplesFactory): The relations used to train the embeddings
        test_data (pykeen.TriplesFactory): The relations used to test the embeddings
        base_folder (str): folder where the pickle should be safed
    """

    el = []
    for _, edge in edges.iterrows():
        if knowledge_graph_type == "quantified_conditions_with_literal" and edge.loc['literal_included'] != 'None':
            continue
        elif knowledge_graph_type == 'basic':
            el.append(str(edge.loc['from']) + "||" + str(edge.loc['to']))
        else:
            el.append(str(edge.loc['from']) + "||" + str(edge.loc['to']))

    G = nx.parse_edgelist(el, delimiter="||", create_using=nx.DiGraph).to_directed()

    closness_centrality = nx.closeness_centrality(G).values()
    degree_centrality = nx.degree_centrality(G).values()
    avgnbdeg = nx.average_neighbor_degree(G, source="in+out").values()
    
    if "_with_literal" in knowledge_graph_type:
        nolit = metadata.loc[metadata['type'] == 'value'].shape[0]
        
    else:
        nolit = 0

    degrees = G.degree()

    sum_of_edges = sum(dict(degrees).values())
    avg_degree = sum_of_edges / G.number_of_nodes()

    relations = {
        r for _, r, _ in np.concatenate((train_data.triples, test_data.triples))
    }

    graph_data = {
        'representation': knowledge_graph_type,
        'No. of edges': len(el),
        'No. of nodes': len(metadata) - nolit,
        'No. of literals': nolit,
        'No. of relations': len(relations),
        'closeness centrality': str(round(mean(closness_centrality), 2)) + "±" + str(round(stdev(closness_centrality), 2)),
        'degree centrality': str(round(mean(degree_centrality), 2)) + "±" + str(round(stdev(degree_centrality), 2)),
        'avg. nb. degree': str(round(mean(avgnbdeg), 2)) + "±" + str(round(stdev(avgnbdeg), 2)),
        'avg. degree': avg_degree
    }
    with open(base_folder + knowledge_graph_type + ".pkl", 'wb') as out_f:
        pickle.dump(graph_data, out_f)