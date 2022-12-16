import experiment_definition as ed
import knowledge_graph as kg
# def weight_and_filter(returned_graphs, experiment_series, weight_methods, filter_methods):
from influences_visualisation import InfluenceVisualiser, create_directories_if_necessary
from data_provider.abstract_data_provider import AbstractDataProvider
from data_provider import data_provider_singleton

def generate_presentation_plots(dp: AbstractDataProvider):
    data, lok, lov, boolean_parameters, returned_graphs, experiment_series, label_encoder = dp.get_experiments_with_graphs(
        influential_influences_only=True)

    definition = ed.ExperimentDefinition(levels=[],
                                         weight_methods=[ed.WeightingClustering.CLUSTERING_PARAMETERS_QUALITY], filter_methods=[ed.AdaptiveFilters.FIRST_QUARTILE])

    # filtered_graphs = weight_and_filter(graphs, experiment_series, definition.weight_methods, definition.filter_methods)
    knowledge_graph = kg.aggregate_unfiltered(returned_graphs)

    weighted_graph = InfluenceVisualiser.weight_clustering(kg.aggregate_unfiltered(returned_graphs), experiment_series, definition.weight_methods[0],
                                                           active_fill=False)
    filtered_graph = InfluenceVisualiser.filter_graph(weighted_graph, definition.filter_methods[0])

    filename = 'figs/presentation_aggregated_full.pdf'
    create_directories_if_necessary(filename)
    InfluenceVisualiser.plot_graph(knowledge_graph, fname=filename)

    InfluenceVisualiser.plot_graph(filtered_graph, 'figs/presentation_filtered.pdf')

    InfluenceVisualiser.plot_graph(kg.filter_knowledge_graph_for_influence(filtered_graph,'stringing'), 'figs/presentation_filtered_stringing.pdf')

    # InfluenceVisualiser.plot_all_envfactor_parameter_pairs_relative(filtered_graph, experiment_series, 'stringing')
    es_ps = kg.extract_e_p_values(filtered_graph, experiment_series,
                                  fill_nones=False, best_only=False)
    InfluenceVisualiser.plot_envfactor_parameter_pair_relative_influence(es_ps['retraction_amount-stringing'],
                                                                         'retraction_amount', 'stringing',
                                                                         relative_parameter=True)


    # plot them relative to previous experiment TODO - why are relative parameter values sometimes 0?!
    # relative_parameters = InfluenceVisualiser._dataframe_preparation(filtered_graph, experiment_series, ed.WeightingClustering.CLUSTERING_PARAMETER_RELATIVE, active_fill=False)
    # relative_qualities = InfluenceVisualiser._dataframe_preparation(filtered_graph, experiment_series, ed.WeightingClustering.CLUSTERING_QUALITY_RELATIVE, active_fill=False)
    #
    # InfluenceVisualiser._plot_ep_detail('retraction_amount', relative_parameters, 'stringing', relative_qualities)

if __name__ == '__main__':
    
    dp = data_provider_singleton.get_data_provider('remote')
    generate_presentation_plots(dp)
