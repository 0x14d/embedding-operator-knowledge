from __future__ import annotations

import logging
from collections import defaultdict
import pickle
from typing import Any, Iterable, Mapping
from tqdm import trange

import igraph
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from data_provider import data_provider_singleton

import knowledge_graph as kg
from data_provider.abstract_data_provider import AbstractDataProvider
from evaluation_visualization_utilities import visualize_performance_of_metric
from rule_evaluation_utilities import evaluate_rules, write_results_to_excel, write_rules_to_file
from experiment_definition import ExperimentDefinition, AbstractionLevel, WeightDuringAggregation, AdaptiveFilters, \
    GroundTruthIdentifier, MetricIdentifier, VisualisationIdentifier, WeightingClustering, WeightingInfluenceValidity, \
    DoubleWeightingClustering, WeightingMethod
from preprocessing import LabelEncoderForColumns
import rule_base.rule_extraction as re
from rule_base.rule_serializer import RuleSerializer
from . import graph_aggregation as graph_agg


class KCAPEvaluation:

    @staticmethod
    def method_comparison(exp_def: ExperimentDefinition,
                          list_of_graphs: Iterable[igraph.Graph],
                          experiment_series: dict,
                          edges: Mapping,
                          _boolean_parameters: Iterable[str],
                          string_parameters: Iterable[str],
                          gt_label_encoder: LabelEncoderForColumns,
                          _label_encoder: LabelEncoderForColumns,
                          viz: bool = True,
                          fname=""):
        """
        created filtered knowledge graphs and evaluates them against different ground truths
        If desired visualization of the filtered graphs is done --> scatter plots with quality changes related to the
        parametrisation or which relations were filtered at all --> define in the experiment definition (constructor)
        """

        params_dict = dict()
        aggregated_graphs = graph_agg.GraphAggregation.get_aggregated_graphs(
            list_of_graphs, exp_def.weight_methods, exp_def.filter_methods,
            experiment_series, edges)

        base_graph = kg.aggregate_unfiltered(list_of_graphs)  # base line
        params_dict['base_graph'] = base_graph  # unfiltered graph
        _ground_truths = dict()
        for truth in exp_def.ground_truths:
            # the functions for loading the ground truth should return the rules at the first place
            _ground_truths[truth.name] = truth(
                label_encoder=gt_label_encoder)[0]

        return_stats = dict()
        # enables TQDM progress bars to work with logging outputting to console
        with logging_redirect_tqdm():

            for ground_truth_name, ground_truth in (pbar_gt := tqdm(
                    _ground_truths.items())):
                # TQDM progress bar wrapped around ground truths
                pbar_gt.set_description(f"Ground Truth  {ground_truth_name}")

                overall_statistics = defaultdict(lambda: defaultdict(dict))
                for rule_extraction_method in (pbar_rem := tqdm(
                        exp_def.rule_extraction_method)):
                    # TQDM progress bar wrapped around rule extraction methods
                    rule_extraction_method: re.RuleExtractionMethod
                    rem_name = str(rule_extraction_method)

                    pbar_rem.set_description(f"Rule Extraction Method  {rem_name}")

                    baseline_rules = re.kg_to_improvement_rules(
                        base_graph,
                        experiment_series,
                        _boolean_parameters,
                        string_parameters,
                        rule_extraction_method=rule_extraction_method,
                        label_encoder=_label_encoder)
                    write_rules_to_file(
                        fname=
                        f'{ground_truth_name.lower()}_{rem_name}_baseline#none',
                        ruleset=baseline_rules,
                        version=RuleSerializer.Version.QUANTIFIED_INFLUENCES)

                    for level in exp_def.abstraction_level:
                        level_name = level.name.lower()

                        baseline_evaluated = evaluate_rules(
                            ground_truth, baseline_rules, level,
                            exp_def.metrics)
                        overall_statistics[rem_name][level_name].update(
                            {'baseline#none': baseline_evaluated.copy()})
                        for method, aggregated_graph in aggregated_graphs.items(
                        ):
                            logging.info(
                                f"evaluation | rule extraction {rem_name} | level: {level_name}\t method: {method} - {ground_truth_name}"
                            )
                            method_rules = re.kg_to_improvement_rules(
                                aggregated_graph['graph'],
                                experiment_series=aggregated_graph[
                                    'exp_series'],
                                boolean_parameters=_boolean_parameters,
                                string_parameters=string_parameters,
                                rule_extraction_method=rule_extraction_method,
                                label_encoder=_label_encoder)
                            params_dict['filtered_graph'] = aggregated_graph[
                                'graph']
                            params_dict['exp_series'] = aggregated_graph[
                                'exp_series']
                            write_rules_to_file(
                                fname=
                                f'{ground_truth_name.lower()}_{rem_name}_{method.lower()}',
                                ruleset=method_rules,
                                version=RuleSerializer.Version.
                                QUANTIFIED_INFLUENCES)
                            method_evaluated = evaluate_rules(
                                ground_truth, method_rules, level,
                                exp_def.metrics)
                            overall_statistics[rem_name][level_name].update(
                                {method: method_evaluated.copy()})
                            if len(exp_def.visualization_methods
                                   ) > 0 and viz is True:
                                KCAPEvaluation._visualize_results(
                                    exp_def, params_dict)

                method_substitutions = {
                    f'{WeightingClustering.CLUSTERING_PARAMETERS_QUALITY.name.lower()}#{AdaptiveFilters.MEAN.name.lower()}':
                    '$C\, p\, q\ \#\ \\bar{x}$',
                    f'{WeightDuringAggregation.CONFIDENCE_QUALITY_IMPROVEMENT_FREQUENCY.name.lower()}#{AdaptiveFilters.MEAN.name.lower()}':
                    '$\zeta\, F\,\Delta q\ \#\ \\bar{x}$',
                    f'{DoubleWeightingClustering.DOUBLECLUSTERING_PARAMETER_QUALITY_NONRELATIVE.name.lower()}#{AdaptiveFilters.FIRST_QUARTILE.name.lower()}':
                    '$CC\, p\, q\ \#\ Q_1$',
                    f'{WeightingInfluenceValidity.CLUSTERING_INFLUENCE_VALIDITY_NONRELATIVE.name.lower()}#{AdaptiveFilters.FIRST_QUARTILE.name.lower()}':
                    '$IV\, \\alpha\, q\ \#\ Q_1$',
                    f'baseline#none':
                    'baseline \cite{Nordsieck.2021}',
                }
                write_results_to_excel(f'{ground_truth_name.lower()}_{fname}',
                                       overall_statistics,
                                       method_substitutions)
                return_stats[ground_truth_name] = overall_statistics
        return return_stats

    @staticmethod
    def _visualize_results(exp_def: ExperimentDefinition, params_dict: dict):
        """
        visualize results of paper evaluation
        :param exp_def experiment definition
        :param params_dict dictionary that will be given to the visualization methods
        Necessary keys depend on the method itself --> see in their doc string
        """
        for visualization in exp_def.visualization_methods:
            if visualization == VisualisationIdentifier.RELATIVE_INFLUENCE_PARAMETER_RELATIVE:
                params_dict['relative_parameter'] = True
            else:
                params_dict['relative_parameter'] = False

            if visualization == VisualisationIdentifier.GRAPH:
                params_dict['file_name'] = "./graph.pdf"

            visualization(params_dict)

    @staticmethod
    def visualize_amount_of_data_needed_for_stable_result(
            exp_def: ExperimentDefinition,
            _label_encoder: LabelEncoderForColumns,
            _dp: AbstractDataProvider,
            min_exp: int = 50,
            max_exp: int = 420,
            step_size: int = 20,
            plot_baseline: bool = False):
        """
        creates a plot that shows you how many experiments are needed to be evaluated to get a stable result for the
        evaluation against the ground truths

        :param exp_def number of experiments that should be added every evaluation step
        :param _label_encoder: pretrained label_encoder on the whole set - otherwise evaluation won't work bc unknown
        labels are encountered
        :param min_exp minimum number of experiments that should be evaluated
        :param max_exp maximum number of experiments that should be evaluated
        :param step_size number of experiments that should be added every evaluation step
        :param plot_baseline number of experiments that should be added every evaluation step
        """

        results = dict()
        with logging_redirect_tqdm():

            for number_exp in (pbar_exp := trange(min_exp, max_exp,
                                                  step_size)):
                pbar_exp.set_description("Num Experiments %f" % number_exp)

                _, _, _lov, _boolean_parameters, _returned_graphs, experiment_series, _limited_label_encoder = \
                    _dp.get_experiments_with_graphs(
                        influential_influences_only=True, limit=number_exp)
                string_parameters = [k[:-1] for k in _lov.keys()]
                _stats = KCAPEvaluation.method_comparison(exp_def,
                                                          _returned_graphs,
                                                          experiment_series,
                                                          _dp.get_edges_dict(),
                                                          _boolean_parameters,
                                                          string_parameters,
                                                          _label_encoder,
                                                          _label_encoder,
                                                          viz=False)
                for ground_truth, rule_extractions in _stats.items():
                    if ground_truth not in results:
                        results[ground_truth] = dict()
                    for rem_name, levels in rule_extractions.items():
                        if rem_name not in results[ground_truth]:
                            results[ground_truth][rem_name] = dict()
                        rem_results = results[ground_truth][rem_name]
                        for level in levels.keys():  # high, mid, low
                            for evaluation_method in levels[level].keys():
                                for metric in levels[level][evaluation_method]:
                                    key = f'{evaluation_method}/{metric}'
                                    if key not in rem_results:
                                        rem_results[key] = dict()
                                    if level not in rem_results[key]:
                                        rem_results[key][level] = dict()
                                        rem_results[key][level]['x'] = list()
                                        rem_results[key][level]['y'] = list()
                                    rem_results[key][level]['x'].append(
                                        number_exp)
                                    rem_results[key][level]['y'].append(
                                        levels[level][evaluation_method]
                                        [metric])
        with logging_redirect_tqdm():
            for _ground_truths in (pbar := tqdm(results)):
                pbar.set_description("Processing %s" % str(_ground_truths))

                for _rem in results[_ground_truths]:
                    for _metrics in results[_ground_truths][_rem]:
                        filename = f'kcap/{_ground_truths}/{_rem}/'
                        _data = dict()
                        key_baseline = [
                            key for key in results[_ground_truths][_rem]
                            if 'baseline' in key
                        ]
                        _data['metric_data'] = results[_ground_truths][_rem][
                            _metrics]
                        _data['baseline'] = results[_ground_truths][_rem][
                            key_baseline[0]]
                        visualize_performance_of_metric(
                            fn=filename + _metrics + '.pdf',
                            data=_data,
                            with_curve_fit=True,
                            metric=_metrics,
                            plot_baseline=plot_baseline)

    @staticmethod
    def setup_rule_extraction_functions() -> list[re.RuleExtractionMethod]:

        def setup_linear_pipeline_steps() -> dict[str, list[tuple[str, Any]]]:
            from sklearn import preprocessing
            import sklearn.linear_model as lm

            pipeline_steps = {
                "LINEAR": [('standardize', preprocessing.StandardScaler()),
                        ('poly', preprocessing.PolynomialFeatures(1)),
                        ('regressor', lm.LinearRegression())],
                "POLY2": [('standardize', preprocessing.StandardScaler()),
                        ('poly', preprocessing.PolynomialFeatures(2)),
                        ('regressor', lm.LinearRegression())],
                "POLY3": [('standardize', preprocessing.StandardScaler()),
                        ('poly', preprocessing.PolynomialFeatures(3)),
                        ('regressor', lm.LinearRegression())],
            }
            return pipeline_steps

        re_pipeline_steps = setup_linear_pipeline_steps()
        re_funs = [
            re.BinnedInfluencesFunction(pipeline_steps=pipe, pipeline_name=name)
            for name, pipe in re_pipeline_steps.items()
        ]
        return re_funs


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description="run KCAP evaluation using the coded experiment definition."
    )
    parser.add_argument(
        '-dp',
        metavar='--dataprovider',
        type=str,
        help="which kind of DataProvider source is to be used.",
        choices=["local", "remote", "synthetic"],
        default="local")

    args = parser.parse_args()
    dp_kind = args.dp

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%d-%m %H:%M',
        filename=f'kcap_comparison_{dp_kind}.log',
        filemode='a')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.info("### Starting KCAP Evaluation Script ###")
    # hide sklearn warnings
    import warnings

    # TODO XAI-632: don't filter warnings, remove their source!
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)




    dp = data_provider_singleton.get_data_provider(dp_kind)
    logging.info(f"created dataprovider of kind {dp_kind}.")
    data, lok, lov, boolean_parameters, returned_graphs, experiment_series_main, syn_label_encoder = dp.get_experiments_with_graphs(
        influential_influences_only=True)
    edges_dict = dp.get_edges_dict()

    # this is a sample experiment definition
    abstraction_levels = [
        AbstractionLevel.HIGH, AbstractionLevel.MID, AbstractionLevel.LOW
    ]
    filter_functions = [
        AdaptiveFilters.MEAN, AdaptiveFilters.MEDIAN,
        AdaptiveFilters.FIRST_QUARTILE
    ]
    weighting_methods: list[WeightingMethod] = [
        WeightDuringAggregation.CONFIDENCE,
        WeightDuringAggregation.FREQUENCY,
        WeightDuringAggregation.QUALITY_IMPROVEMENT,
        WeightDuringAggregation.CONFIDENCE_FREQUENCY,
        WeightDuringAggregation.CONFIDENCE_QUALITY_IMPROVEMENT,
        WeightDuringAggregation.CONFIDENCE_QUALITY_IMPROVEMENT_FREQUENCY,
        WeightDuringAggregation.FREQUENCY_QUALITY_IMPROVEMENT,
        WeightingClustering.CLUSTERING_ALL,
        WeightingClustering.CLUSTERING_NON_RELATIVE,
        WeightingClustering.CLUSTERING_ALL_RELATIVE,
        WeightingClustering.CLUSTERING_PARAMETER_RELATIVE,
        WeightingClustering.CLUSTERING_INFLUENCES,
        WeightingClustering.CLUSTERING_INFLUENCES_RELATIVE,
        WeightingClustering.CLUSTERING_PARAMETERS_QUALITY,
        WeightingClustering.CLUSTERING_QUALITY_RELATIVE,
        WeightingInfluenceValidity.CLUSTERING_INFLUENCE_VALIDITY_NONRELATIVE,
        WeightingInfluenceValidity.CLUSTERING_INFLUENCE_VALIDITY_RELATIVE,
        DoubleWeightingClustering.
        DOUBLECLUSTERING_PARAMETER_QUALITY_NONRELATIVE,
        DoubleWeightingClustering.DOUBELCLUSTERING_PARAMETER_QUALITY_RELATIVE,
    ]
    relevant_weighting_methods: list[WeightingMethod] = [
        # WeightDuringAggregation.CONFIDENCE_QUALITY_IMPROVEMENT_FREQUENCY,
        # WeightingClustering.CLUSTERING_ALL,
        WeightingClustering.CLUSTERING_PARAMETERS_QUALITY,
        # WeightingInfluenceValidity.CLUSTERING_INFLUENCE_VALIDITY_NONRELATIVE,
        # DoubleWeightingClustering.DOUBLECLUSTERING_PARAMETER_QUALITY_NONRELATIVE,
    ]

    relevant_filter_methods = [
        AdaptiveFilters.MEAN, AdaptiveFilters.FIRST_QUARTILE
    ]
    ground_truths = [
        # GroundTruthIdentifier.KCAP_EXPERT_LABELLED,
        GroundTruthIdentifier.V3_EXPERT_LABELLED,
    ]
    metrics = [
        MetricIdentifier.PRECISION, MetricIdentifier.RECALL,
        MetricIdentifier.F1, MetricIdentifier.AMOUNT_RULES
    ]
    visualization_methods = []


    rule_extraction_methods = [
        re.FromEdge,
        re.BinnedInfluences,
        re.BinnedMergedInfluences,
        *KCAPEvaluation.setup_rule_extraction_functions()
    ]

    experiment_def = ExperimentDefinition(
        levels=abstraction_levels,
        filter_methods=filter_functions,
        weight_methods=weighting_methods,
        ground_truths=ground_truths,
        metrics=metrics,
        visualization_methods=visualization_methods,
        rule_extraction_methods=rule_extraction_methods)

    relevant_experiment_def = ExperimentDefinition(
        levels=abstraction_levels,
        filter_methods=relevant_filter_methods,
        weight_methods=relevant_weighting_methods,
        ground_truths=ground_truths,
        metrics=metrics,
        visualization_methods=visualization_methods,
        rule_extraction_methods=rule_extraction_methods)

    if dp_kind == "synthetic":
        # the local AIPE DP-Label encoder needs to be used, since it was used for creating the ground truth. should be versioned with the ground truth data in the future!
        _, _, _, _, _, _, gt_label_encoder = data_provider_singleton._get_local_aipe_dp().get_experiments_with_graphs( influential_influences_only=True)
    else:
        gt_label_encoder = data_provider_singleton.get_label_encoder(dp_kind)


    from datetime import datetime
    date = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

    filename = f"{dp_kind}_{date}"

    logging.info("got experiments from dp, starting kcap comparison.")
    stats = KCAPEvaluation.method_comparison(relevant_experiment_def,
                                             returned_graphs,
                                             experiment_series_main,
                                             edges_dict,
                                             boolean_parameters,
                                             [k[:-1] for k in lov.keys()],
                                             gt_label_encoder,
                                             syn_label_encoder,
                                             fname=filename)
    # import pickle
    # with open(f"kcap_results_{date}.pkl", 'wb') as pkl:
    #     pickle.dump(stats, pkl)

    # logging.info("finished kcap method comparison. starting visualization comparison")
    # visualisation_def = ExperimentDefinition(levels=[
    #     AbstractionLevel.HIGH,
    #     AbstractionLevel.MID,
    #     AbstractionLevel.LOW],
    #     filter_methods=[AdaptiveFilters.MEAN], weight_methods=[
    #     WeightingClustering.CLUSTERING_PARAMETERS_QUALITY], ground_truths=ground_truths, metrics=[MetricIdentifier.F1], visualization_methods=visualization_methods, rule_extraction_methods=rule_extraction_methods)

    # KCAPEvaluation.visualize_amount_of_data_needed_for_stable_result(visualisation_def, label_encoder, plot_baseline=True)
