"""Module that provied functionality to import a knowledge graph"""

# pylint: disable=import-error

import os
from typing import Optional, Tuple

import igraph
import pandas as pd

from data_provider.abstract_data_provider import AbstractDataProvider
from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorConfig
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_provider import SyntheticDataProvider
from knowledge_extraction.rule_to_representation import get_graph_from_data_provider

def import_knowledge_graph(
    influential_only: bool,
    data_provider: AbstractDataProvider,
    directory: Optional[str] = None,
    sdg_config: Optional[SdgConfig] = None,
    kg_config: Optional[KnowledgeGraphGeneratorConfig] = None
) -> Tuple[igraph.Graph, Optional[pd.DataFrame]]:
    """
    Imports a knowledge graph.
    If directory is defined it will be tried to import the kg from the files.
    Otherwise or if the file import fails the kg will be imported from the dataprovider.

    Parameters:
        - influential_only (bool): defines if only influencial experiments should be fetched
        - data_provider (AbstractDataProvider): dataprovider that is / was used to import the kg
        - directory (str, optional): directory that contains the kg
        - sdg_config (SdgConfig, optional): sdg config used for the synthetic data provider
        - kg_config (KnowledgeGraphGeneratorConfig, optional):
            config of the kg generator used to create the kg

    Returns:
        imported knowledge graph,
        changed parameters (if kg is loaded from the AIPE database, else None)
    """
    args = {
        'directory': directory,
        'influential_only': influential_only,
        'data_provider': data_provider
    }
    try:
        if directory is None:
            raise ValueError()
        return _load_kg_from_files(*args)
    except (FileNotFoundError, ValueError):
        return _load_kg_from_dataprovider(
            **args,
            sdg_config=sdg_config,
            kg_config=kg_config
        ), None

def _load_kg_from_files(
    directory: str,
    influential_only: bool,
    data_provider: AbstractDataProvider
) -> Tuple[igraph.Graph, pd.DataFrame]:
    """Loads the kg from the specified directory"""
    if influential_only:
        file_name_add = '_inf_only.pkl'
    else:
        file_name_add = '.pkl'
    edges = pd.read_pickle(directory + 'graph_embeddings/edges' + file_name_add)
    vertecies = pd.read_pickle(directory + 'graph_embeddings/verts' + file_name_add)
    if not isinstance(data_provider, SyntheticDataProvider):
        file = directory + 'graph_embeddings/parameters' + file_name_add
        changed_parameters = pd.read_pickle(file)
    graph = igraph.Graph.DataFrame(edges, directed=True, vertices=vertecies)

    return graph, changed_parameters


def _load_kg_from_dataprovider(
    directory: Optional[str],
    influential_only: bool,
    data_provider: AbstractDataProvider,
    sdg_config: Optional[SdgConfig] = None,
    kg_config: Optional[KnowledgeGraphGeneratorConfig] = None
) -> igraph.Graph:
    """Loads the kg from the specified dataprovider"""
    print("Import Graph Data from Database")
    graph = get_graph_from_data_provider(
        data_provider=data_provider,
        sdg_config=sdg_config,
        kg_config=kg_config,
        influential_only=influential_only
    )

    # Save kg data
    if directory is not None:
        df_ed = graph.get_edge_dataframe()
        df_vs = graph.get_vertex_dataframe()
        if influential_only:
            file_name_add = '_inf_only.pkl'
        else:
            file_name_add = '.pkl'
        if not os.path.isdir(directory + 'graph_embeddings/'):
            os.makedirs(directory + 'graph_embeddings/')
        if not os.path.exists(directory + 'graph_embeddings/edges' + file_name_add):
            df_ed.to_pickle(directory + 'graph_embeddings/edges' + file_name_add)
        if not os.path.exists(directory + 'graph_embeddings/verts' + file_name_add):
            df_vs.to_pickle(directory + 'graph_embeddings/verts' + file_name_add)

    return graph
