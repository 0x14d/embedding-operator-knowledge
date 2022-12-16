# pylint: disable=import-error

from typing import Optional
import igraph

from rule_base.rule_extraction import kg_to_improvement_rules
from data_provider.knowledge_graphs.generators.abstract_knowledge_graph_generator \
    import KnowledgeGraphGenerator
from data_provider.synthetic_data_provider import SyntheticDataProvider
from data_provider.abstract_data_provider import AbstractDataProvider
from data_provider.knowledge_graphs.pq_relation import PQ_Relation
from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorConfig
from data_provider.synthetic_data_generation.types.generator_arguments \
    import KnowledgeGraphGeneratorArguments
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from knowledge_aggregation import kg

def get_graph_from_data_provider(
    data_provider: AbstractDataProvider,
    sdg_config: Optional[SdgConfig] = None,
    kg_config: Optional[KnowledgeGraphGeneratorConfig] = None,
    influential_only: bool = True
) -> igraph.Graph:
    """Gets the graph from the dataprovider

    Args:
        data_provider (AbstractDataProvider): the data provider
        sdg_config (SdgConfig, optional): sdg config used for the synthetic data provider
        kg_config (KnowledgeGraphGeneratorConfig, optional): config of the kg generator used to create the kg
        influential_only (bool, optional, default=True):
            determines if only influential experiments should be fetched from the dp

    Returns:
        iGraph.Graph: KnowledgeGraph
    """
    if isinstance(data_provider, SyntheticDataProvider):
        return data_provider.data_generator.knowledge_graph

    _, _, lov, boolean_parameters, returned_graphs, experiment_series, _ = \
        data_provider.get_experiments_with_graphs(
            influential_influences_only=influential_only
        )
    string_parameters = [k[:-1] for k in lov.keys()]

    graph = kg.aggregate_unfiltered(returned_graphs)
    rules = kg_to_improvement_rules(
        knowledge_graph=graph,
        experiment_series=experiment_series,
        boolean_parameters=boolean_parameters,
        string_parameters=string_parameters,
        dp = data_provider
    )
    rels = [PQ_Relation.from_rule(rule) for rule in rules]

    if sdg_config is None:
        sdg_config = SdgConfig.create_config(
            'knowledge_infusion/eval_with_synth_data/configs/sdg/default_config_sdg.json'
        )
    if kg_config is not None:
        sdg_config.knowledge_graph_generator = kg_config

    knowledge_graph_generator: KnowledgeGraphGenerator = sdg_config.knowledge_graph_generator.get_generator_class()(
        KnowledgeGraphGeneratorArguments(
                sdg_config=sdg_config,
                pq_functions=None,
                pq_tuples=None,
                pq_relations=rels
        )
    )
    return knowledge_graph_generator.generate_knowledge_graph()


if __name__ == '__main__':
    # pylint: disable=ungrouped-imports
    from data_provider import data_provider_singleton
    data_provider_class = data_provider_singleton.get_data_provider("remote")
    get_graph_from_data_provider(data_provider_class)
