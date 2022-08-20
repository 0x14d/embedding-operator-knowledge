from enum import Enum
from functools import partial
from typing import Union

import numpy as np
import pandas
import sklearn
from matplotlib import pyplot as plt
from scipy.spatial import distance
import sklearn.pipeline

import evaluation_visualization_utilities
import rule_evaluation_utilities
from rule_base import rule
from rule_base.rule_extraction import RuleExtractionMethod
from rule_base.rule_serializer import RuleSerializer
from influences_visualisation import InfluenceVisualiser


class ConfigBase(Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class Metrics:
    @staticmethod
    def jaccard(ground_truth, prediction, len_prediction):
        return sklearn.metrics.jaccard_score(ground_truth, prediction)

    @staticmethod
    def jensenshannon_divergence(ground_truth, prediction, len_prediction):
        # divergence = distance squared
        return distance.jensenshannon(ground_truth, prediction) ** 2

    @staticmethod
    def cross_entropy(ground_truth, prediction, len_prediction):
        return sklearn.metrics.log_loss(ground_truth, prediction, labels=[1, 0])

    @staticmethod
    def f1_normalized_by_length(ground_truth, prediction, len_prediction):
        return sklearn.metrics.f1_score(ground_truth, prediction) / len(ground_truth)

    @staticmethod
    def f1_score(ground_truth, prediction, len_prediction):
        return sklearn.metrics.f1_score(ground_truth, prediction)

    @staticmethod
    def f1_normalized_by_false_pred(ground_truth, prediction, len_prediction):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            ground_truth, prediction).ravel()
        return sklearn.metrics.f1_score(ground_truth, prediction) / (fn + fp)

    @staticmethod
    def js_normalized_by_false_pred(ground_truth, prediction, len_prediction):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            ground_truth, prediction).ravel()
        return Metrics.jensenshannon_divergence(ground_truth, prediction, len_prediction) / (fn + fp)

    @staticmethod
    def precision_normalized_by_false_pred(ground_truth, prediction, len_prediction):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            ground_truth, prediction).ravel()
        return sklearn.metrics.precision_score(ground_truth, prediction) / (fn + fp)

    @staticmethod
    def precision_score(ground_truth, prediction, len_prediction):
        return sklearn.metrics.precision_score(ground_truth, prediction)

    @staticmethod
    def recall_normalized_by_false_pred(ground_truth, prediction, len_prediction):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            ground_truth, prediction).ravel()
        return sklearn.metrics.recall_score(ground_truth, prediction) / (fn + fp)

    @staticmethod
    def recall_score(ground_truth, prediction, len_prediction):
        return sklearn.metrics.recall_score(ground_truth, prediction)

    @staticmethod
    def amount_of_rules(ground_truth, prediction, len_prediction):
        """

        :param ground_truth:
        :param prediction:
        :param len_prediction: required as a separate parameter since len of substituted rule sets is likely to be greater than original ruleset
        :return:
        """
        return len_prediction


class WeightDuringAggregationFunctions:
    @staticmethod
    def confidence(confidence, frequency, quality_improvement):
        return confidence

    @staticmethod
    def frequency(confidence, frequency, quality_improvement):
        return frequency

    @staticmethod
    def quality_improvement(confidence, frequency, quality_improvement):
        return quality_improvement

    @staticmethod
    def confidence_frequency(confidence, frequency, quality_improvement):
        return 0.5 * frequency + confidence * frequency

    @staticmethod
    def confidence_quality_improvement(confidence, frequency, quality_improvement):
        return 0.5 * quality_improvement + confidence * quality_improvement

    @staticmethod
    def confidence_quality_improvement_frequency(confidence, frequency, quality_improvement):
        weight = 0.5 * quality_improvement + confidence * quality_improvement
        return weight * frequency

    @staticmethod
    def frequency_quality_improvement(confidence, frequency, quality_improvement):
        return frequency * quality_improvement


class WeightDuringAggregation(ConfigBase):
    CONFIDENCE = partial(WeightDuringAggregationFunctions.confidence)
    FREQUENCY = partial(WeightDuringAggregationFunctions.frequency)
    QUALITY_IMPROVEMENT = partial(
        WeightDuringAggregationFunctions.quality_improvement)
    CONFIDENCE_FREQUENCY = partial(
        WeightDuringAggregationFunctions.confidence_frequency)
    CONFIDENCE_QUALITY_IMPROVEMENT = partial(
        WeightDuringAggregationFunctions.confidence_quality_improvement)
    CONFIDENCE_QUALITY_IMPROVEMENT_FREQUENCY = partial(
        WeightDuringAggregationFunctions.confidence_quality_improvement_frequency)
    FREQUENCY_QUALITY_IMPROVEMENT = partial(
        WeightDuringAggregationFunctions.frequency_quality_improvement)


class WeightingClustering(ConfigBase):
    """
    Weighting strategies for clustering mainly differ by the dataframe that is used as basis for the clustering, additional kwargs (i.e. hyperparamters for the clustering) can be provided.
    """
    CLUSTERING_ALL = partial(lambda filled_dfs: pandas.concat(
        [filled_dfs['changed_parameters'], filled_dfs['changed_parameters_relative'], filled_dfs['quality'],
         filled_dfs['quality_relative'], filled_dfs['influences'], filled_dfs['influences_relative']], axis=1,
        sort=False))
    CLUSTERING_NON_RELATIVE = partial(lambda filled_dfs: pandas.concat(
        [filled_dfs['changed_parameters'], filled_dfs['quality'], filled_dfs['influences']], axis=1, sort=False))
    CLUSTERING_ALL_RELATIVE = partial(lambda filled_dfs: pandas.concat(
        [filled_dfs['changed_parameters_relative'], filled_dfs['quality_relative'],
         filled_dfs['influences_relative']], axis=1, sort=False))
    CLUSTERING_PARAMETER_RELATIVE = partial(
        lambda dfs: dfs['changed_parameters_relative'])
    CLUSTERING_INFLUENCES = partial(lambda filled_dfs: pandas.concat(
        [filled_dfs['influences']], axis=1,
        sort=False))
    CLUSTERING_INFLUENCES_RELATIVE = partial(lambda filled_dfs: pandas.concat(
        [filled_dfs['influences_relative']], axis=1,
        sort=False))
    CLUSTERING_PARAMETERS_QUALITY = partial(lambda dfs: pandas.concat([dfs['changed_parameters'], dfs['quality']],
                                                                      axis=1, sort=False))

    CLUSTERING_PARAMETERS_INFLUENCES = partial(lambda dfs: pandas.concat([dfs['influences'], dfs['quality']],
                                                                         axis=1, sort=False))
    CLUSTERING_QUALITY_RELATIVE = partial(lambda dfs: dfs['quality_relative'])


class DoubleWeightingClustering(ConfigBase):
    DOUBELCLUSTERING_PARAMETER_QUALITY_RELATIVE = partial(
        lambda filled_dfs: (filled_dfs['changed_parameters_relative'], filled_dfs['quality_relative']))
    DOUBLECLUSTERING_PARAMETER_QUALITY_NONRELATIVE = partial(
        lambda filled_dfs: (filled_dfs['changed_parameters'], filled_dfs['quality']))


class WeightingInfluenceValidity(ConfigBase):
    """
    Weighting strategies for relation based weighting, like their clustering counterparts, are only separated by the df they are run on, i.e. whether they take absolute or relative quality change into consideration.
    """
    CLUSTERING_INFLUENCE_VALIDITY_RELATIVE = partial(
        lambda filled_dfs: (filled_dfs['quality_relative'], filled_dfs['influences_relative']))
    CLUSTERING_INFLUENCE_VALIDITY_NONRELATIVE = partial(
        lambda filled_dfs: (filled_dfs['quality'], filled_dfs['influences']))


class AdaptiveFilters(ConfigBase):
    MEAN = partial(np.mean)
    MEDIAN = partial(np.median)
    FIRST_QUARTILE = partial(lambda weights: plt.boxplot(weights)[
                             'boxes'][0].get_ydata()[0])


class MetricIdentifier(ConfigBase):
    JENSEN_SHANNON = partial(Metrics.jensenshannon_divergence)
    CROSS_ENTROPY = partial(Metrics.cross_entropy)
    F1_NORMALIZED_BY_LENGTH = partial(Metrics.f1_normalized_by_length)
    F1_NORMALIZED_BY_FALSE_PRED = partial(Metrics.f1_normalized_by_false_pred)
    JS_NORMALIZED_BY_FALSE_PRED = partial(Metrics.js_normalized_by_false_pred)
    F1 = partial(Metrics.f1_score)
    PRECISION = partial(Metrics.precision_score)
    PRECISION_NORMALIZED_BY_FALSE_PRED = partial(
        Metrics.precision_normalized_by_false_pred)
    RECALL = partial(Metrics.recall_score)
    RECALL_NORMALIZED_BY_FALSE_PRED = partial(
        Metrics.recall_normalized_by_false_pred)
    JACCARD = partial(Metrics.jaccard)
    AMOUNT_RULES = partial(Metrics.amount_of_rules)


class AbstractionLevel(ConfigBase):
    HIGH = partial(rule.Rule.highlevel_eq)
    MID = partial(rule.Rule.midlevel_eq)
    LOW = partial(rule.Rule.lowlevel_eq)


class GroundTruthIdentifier(ConfigBase):
    KCAP_EXPERT_LABELLED = partial(rule_evaluation_utilities.load_expert_survey_baseline_rules,
                                   dirname='surveys/kcap',
                                   version=RuleSerializer.Version.KCAP)
    V3_EXPERT_LABELLED = partial(rule_evaluation_utilities.load_expert_survey_baseline_rules,
                                 dirname='surveys/v3',
                                 version=RuleSerializer.Version.VERSION3)


class VisualisationIdentifier(ConfigBase):
    """
    These functions have a standardized interface in which only a params_dict and additional default parameters are
    allowed.
    The keys needed in this dictionary can depend on the function itself but some keys are for now predefined:
    'base_graph' --> <iGraph.Graph> unfiltered aggregated knowledge graph
    'filtered_graph' --> <iGraph.Graph> weighted and filtered aggregated knowledge graph
    'exp_series' --> <dict> experiment_series of the filtered graph
    'relative_parameter' --> <bool> True if the x-axis of the scatter plot (only relevant for RELATIVE_INFLUENCE and
    RELATIVE_INFLUENCE_PARAMETER_RELATIVE) should display the change of parameter against the relative quality change
    'fname' --> <string> possible filename (currently only used in GRAPH)
    """
    GRAPH_COMPARISON = partial(evaluation_visualization_utilities.visualize_graph_filtering)
    GRAPH = partial(InfluenceVisualiser.plot_graph)
    RELATION = partial(InfluenceVisualiser.plot_envfactor_parameter_pair)
    RELATIVE_INFLUENCE = partial(
        InfluenceVisualiser.plot_envfactor_parameter_pair_relative_influence)
    RELATIVE_INFLUENCE_PARAMETER_RELATIVE = partial(
        InfluenceVisualiser.plot_envfactor_parameter_pair_relative_influence)
    RELATIVE_INFLUENCE_PRECEEDING_EXP = partial(
        InfluenceVisualiser.plot_preceeding_experiment)


WeightingMethod = Union[WeightingClustering, WeightDuringAggregation, WeightingInfluenceValidity, DoubleWeightingClustering]
class ExperimentDefinition:
    def __init__(self, levels: list[AbstractionLevel],
                 weight_methods: list[WeightingMethod],
                 filter_methods: list[AdaptiveFilters], 
                 metrics: list[MetricIdentifier],
                 ground_truths: list[GroundTruthIdentifier],
                 visualization_methods: list[VisualisationIdentifier],
                 rule_extraction_methods: list[RuleExtractionMethod]):
        self.abstraction_level = levels
        self.weight_methods = weight_methods
        self.filter_methods = filter_methods
        self.metrics = metrics
        self.ground_truths = ground_truths
        self.visualization_methods = visualization_methods
        self.rule_extraction_method = rule_extraction_methods
