# pylint: disable=import-error

from typing import List, Optional
import igraph

from experiment_definition import GroundTruthIdentifier
from rule_base.rule import Rule
from rule_base.rule_extraction import kg_to_improvement_rules, RuleExtractionMethod, FromEdge
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
from knowledge_infusion.graph_embeddings.embedding_config import KnowledgeExtractionMethod
from knowledge_extraction.graph_aggregation import GraphAggregation


def get_graph_from_rules(
    rules: List[Rule],
    sdg_config: SdgConfig,
    kg_config: Optional[KnowledgeGraphGeneratorConfig] = None,
) -> igraph.Graph:
    """
    Creates a knowledge graph from a list of rules


    Args:
        rules (List[Rule]): list of rules that should be represented in the kg
        sdg_config (SdgConfig): sdg config used for the synthetic data provider
        kg_config (KnowledgeGraphGeneratorConfig, optional): config of the kg generator
            used to create the kg

    Returns:
        iGraph.Graph: Created knowledge graph

    """
    rels = [PQ_Relation.from_rule(rule) for rule in rules]
    rels = [r for r in rels if r is not None]

    if sdg_config is None:
        raise ValueError('Sdg config must be provided!')
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


def get_graph_from_data_provider(
    data_provider: AbstractDataProvider,
    sdg_config: SdgConfig,
    kg_config: Optional[KnowledgeGraphGeneratorConfig] = None,
    influential_only: bool = True,
    knowledge_extraction_method: KnowledgeExtractionMethod
        = KnowledgeExtractionMethod.AGGREGATE_UNFILTERED,
    rule_extraction_method: RuleExtractionMethod = FromEdge,
    **kwargs
) -> igraph.Graph:
    """
    Gets the graph from the dataprovider

    Args:
        data_provider (AbstractDataProvider): the data provider
        sdg_config (SdgConfig): sdg config used for the synthetic data provider
        kg_config (KnowledgeGraphGeneratorConfig, optional): config of the kg generator
            used to create the kg
        influential_only (bool, optional, default=True):
            determines if only influential experiments should be fetched from the dp
        knowledge_extraction_method (KnowledgeExtractionMethod, optional,
            default=AGGREGATE_UNFILTERED):
            knowledge extraction method used to generate the kg
        kwargs:
            - knowledge_extraction_weight_function: weight function used for knowledge extraction
            - knowledge_extraction_filter_function: filter function used for knowledge extraction

    Returns:
        iGraph.Graph: KnowledgeGraph
    """
    _, _, lov, boolean_parameters, returned_graphs, experiment_series, label_encoder = \
        data_provider.get_experiments_with_graphs(
            influential_influences_only=influential_only,
            limit=436
        )

    if knowledge_extraction_method == KnowledgeExtractionMethod.GROUNDTRUTH:
        # Load rules from survey
        representation = kg_config.type if kg_config is not None else sdg_config.knowledge_graph_generator.type
        if isinstance(data_provider, SyntheticDataProvider):
            gt_method = GroundTruthIdentifier.SYNTHETIC_V3 if "QUANTIFIED_CONDITIONS" in representation.name else GroundTruthIdentifier.SYNTHETIC_KCAP
            rules, _ = gt_method(label_encoder=label_encoder,
                                 data_provider=data_provider)
        else:
            raise NotImplementedError(
                f'Groundtruth for data provider {data_provider} isn\'t implemented!')

    else:
        # Load rules from knowledge graph
        if knowledge_extraction_method == KnowledgeExtractionMethod.AGGREGATE_UNFILTERED:
            graph = kg.aggregate_unfiltered(returned_graphs)
        elif knowledge_extraction_method == KnowledgeExtractionMethod.AGGREGATE_FILTERED:
            weight_function = kwargs.pop(
                'knowledge_extraction_weight_function')
            filter_function = kwargs.pop(
                'knowledge_extraction_filter_function')
            graph_dict = GraphAggregation.get_aggregated_graphs(
                returned_graphs, [weight_function], [filter_function],
                experiment_series, data_provider.get_edges_dict())
            graph = list(graph_dict.values())[0]['graph']
        else:
            raise NotImplementedError(
                f'Knowledge extraction method {knowledge_extraction_method.value} isn\'t implemented!')
        string_parameters = [k[:-1] for k in lov.keys()]
        rules = kg_to_improvement_rules(
            knowledge_graph=graph,
            experiment_series=experiment_series,
            boolean_parameters=boolean_parameters,
            string_parameters=string_parameters,
            rule_extraction_method=rule_extraction_method,
            dp=data_provider,
            limit=436,
            label_encoder=label_encoder
        )

    return get_graph_from_rules(rules, sdg_config, kg_config)
