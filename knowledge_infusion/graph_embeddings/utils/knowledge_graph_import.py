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
from knowledge_infusion.graph_embeddings.embedding_config import KnowledgeExtractionMethod
from rule_base.rule_extraction import RuleExtractionMethod, FromEdge


def import_knowledge_graph(
    influential_only: bool,
    data_provider: AbstractDataProvider,
    directory: Optional[str] = None,
    sdg_config: Optional[SdgConfig] = None,
    kg_config: Optional[KnowledgeGraphGeneratorConfig] = None,
    knowledge_extraction_method: KnowledgeExtractionMethod \
        = KnowledgeExtractionMethod.AGGREGATE_UNFILTERED,
    rule_extraction_method: RuleExtractionMethod = FromEdge,
    **kwargs
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
        - knowledge_extraction_method (KnowledgeExtractionMethod, optional,
            default=AGGREGATE_UNFILTERED):
            knowledge extraction method used to generate the kg
        - kwargs
            - knowledge_extraction_weight_function: weight function used for knowledge extraction
            - knowledge_extraction_filter_function: filter function used for knowledge extraction

    Returns:
        imported knowledge graph,
        changed parameters (if kg is loaded from the AIPE database, else None)
    """
    args = {
        'directory': directory,
        'influential_only': influential_only,
        'data_provider': data_provider,
        'knowledge_extraction_method': knowledge_extraction_method,
        'rule_extraction_method': rule_extraction_method
    }
    try:
        if directory is None:
            raise ValueError()
        return _load_kg_from_files(**args, **kwargs)
    except (FileNotFoundError, ValueError):
        return _load_kg_from_dataprovider(
            **args,
            sdg_config=sdg_config,
            kg_config=kg_config,
            **kwargs
        ), None


def _load_kg_from_files(
    directory: str,
    influential_only: bool,
    data_provider: AbstractDataProvider,
    knowledge_extraction_method: KnowledgeExtractionMethod,
    rule_extraction_method: RuleExtractionMethod,
    **kwargs
) -> Tuple[igraph.Graph, pd.DataFrame]:
    """Loads the kg from the specified directory"""
    influential_only_add = '_inf_only' if influential_only else ''
    if knowledge_extraction_method == KnowledgeExtractionMethod.AGGREGATE_FILTERED:
        weight_function_add = kwargs['knowledge_extraction_weight_function'].name
        filter_function_add = kwargs['knowledge_extraction_filter_function'].name
        knowledge_extraction_method_add = f'{weight_function_add}_{filter_function_add}'
    else:
        knowledge_extraction_method_add = knowledge_extraction_method.value
    rule_extraction_method_add = f'_{rule_extraction_method.mode.name}'
    file_name_add = f'_{knowledge_extraction_method_add}{rule_extraction_method_add}{influential_only_add}.pkl'
    directory = f'{directory}graph_embeddings/'

    edges = pd.read_pickle(directory + 'edges' + file_name_add)
    vertecies = pd.read_pickle(directory + 'verts' + file_name_add)
    changed_parameters = None
    if not isinstance(data_provider, SyntheticDataProvider):
        file = directory + 'parameters' + file_name_add
        changed_parameters = pd.read_pickle(file)
    graph = igraph.Graph.DataFrame(edges, directed=True, vertices=vertecies)

    return graph, changed_parameters


def _load_kg_from_dataprovider(
    directory: Optional[str],
    influential_only: bool,
    data_provider: AbstractDataProvider,
    sdg_config: Optional[SdgConfig] = None,
    kg_config: Optional[KnowledgeGraphGeneratorConfig] = None,
    knowledge_extraction_method: KnowledgeExtractionMethod \
        = KnowledgeExtractionMethod.AGGREGATE_UNFILTERED,
    rule_extraction_method: RuleExtractionMethod = FromEdge,
    **kwargs
) -> igraph.Graph:
    """Loads the kg from the specified dataprovider"""
    print("Import Graph Data from Database")
    graph = get_graph_from_data_provider(
        data_provider=data_provider,
        sdg_config=sdg_config,
        kg_config=kg_config,
        influential_only=influential_only,
        knowledge_extraction_method=knowledge_extraction_method,
        rule_extraction_method=rule_extraction_method,
        **kwargs
    )
    # Save kg data
    if directory is not None:
        df_ed = graph.get_edge_dataframe()
        df_vs = graph.get_vertex_dataframe()

        influential_only_add = '_inf_only' if influential_only else ''
        if knowledge_extraction_method == KnowledgeExtractionMethod.AGGREGATE_FILTERED:
            weight_function_add = kwargs['knowledge_extraction_weight_function'].name
            filter_function_add = kwargs['knowledge_extraction_filter_function'].name
            knowledge_extraction_method_add = f'{weight_function_add}_{filter_function_add}'
        else:
            knowledge_extraction_method_add = knowledge_extraction_method.value
        file_name_add = f'_{knowledge_extraction_method_add}{influential_only_add}.pkl'
        directory = f'{directory}graph_embeddings/'

        if not os.path.isdir(directory):
            os.makedirs(directory)
        if not os.path.exists(directory + 'edges' + file_name_add):
            df_ed.to_pickle(directory + 'edges' + file_name_add)
        if not os.path.exists(directory + 'verts' + file_name_add):
            df_vs.to_pickle(directory + 'verts' + file_name_add)

    return graph
