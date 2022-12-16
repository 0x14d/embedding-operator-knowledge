""" This module contains a collection of functions to extract Rule objects from a Knowledge Graph's attribute dataframes.

The main function to use is `kg_to_improvement_rules` with one of the globally predefined `RuleExtractionMethod`s (FromEdge, Binned Influences, etc.)

Binning is the discretization of the continuous values of the influences/qualities in to discrete bins. This is carried out using `np.histogram_bin_edges`, the algorithm used to automatically calculate bin edges can be supplied per string, their definitions can be found in the mentioned numpy documentation. Binning is used to reduce the number of possible Rules created from Knowledge Graph Relations.

- `FromEdges` creates one Rule per Parameter-Influence Edge in the Knowledge Graph
- `BinnedInfluences` extracts Rules by first discretizing the values of the influential qualities into bins and then creating a Rule for each bin and each Parameter-Influence pair.
- `BinnedMergedInfluences` works like `BinnedInfluences`, but additionally tries to merge the generate Rules if their relative parameter changes are similiar enough, i.e. they lie within the same discretized bin
- `BinnedInfluencesFunction` tries to fit a (Linear) Regressor or Classifier on the co-occuring parameter-influence data to model their relationship. This fitted function is then sampled at discretized bin intervals to create Rules.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Iterable, Callable, Mapping, MutableMapping, Sequence, Any
from typing import Optional
import warnings

import igraph
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
import sklearn.dummy
from sklearn.exceptions import UndefinedMetricWarning
import sklearn.linear_model as lm
import sklearn.pipeline
import sklearn.preprocessing as prep
import sklearn.svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from data_provider import data_provider_singleton
import knowledge_graph as kg
from preprocessing import LabelEncoderForColumns
from utilities import pairwise
from .rule import Rule


class KgToRuleMode(Enum):
    ''' way of creating Rules from a Knowledge Graph. Do not use this directly, but `RuleExtractionMethod` instead.'''
    FROM_EDGES = 0
    BINNED_INFLUENCES = 1
    BINNED_INFLUENCES_MERGED = 2
    BINNED_INFLUENCES_FUNCTION = 3


@dataclass
class RuleExtractionMethod:
    ''' method to extract Rules from a Knowledge Graph. Optionally include a pipeline for extracting Rules by fitting a Regressor/Classifier pipeline. The last step needs to be an sklearn Estimator.

    `pipeline_steps` and `pipeline_name` are mandatory if `KgToRuleMode.BINNED_INFLUENCES_FUNCTION` is used, in other modes, it will not be used.
    '''
    mode: KgToRuleMode
    pipeline_steps: Optional[list[tuple]] = None
    ''' pipeline steps to be fitted on the PQ-data if mode is `KgToRuleMode.BINNED_INFLUENCES_FUNCTION`. e.g. `[('scaling', StandardScaler()), ('poly', PolynomialFeatures()),('regr',LinearRegression())]`'''
    pipeline_name: Optional[str] = None

    def __str__(self) -> str:
        if self.pipeline_name is not None:
            return f"{self.mode.name}_{self.pipeline_name}".lower()
        else:
            return self.mode.name.lower()

    def make_new_pipeline(self) -> Optional[sklearn.pipeline.Pipeline]:
        ''' create a new cloned copy of the scikit-learn estimator pipeline'''
        if self.pipeline_steps is not None:
            new_steps = [(name, sklearn.base.clone(step))
                         for name, step in self.pipeline_steps]
            return sklearn.pipeline.Pipeline(steps=new_steps)
        else:
            return None


FromEdge = RuleExtractionMethod(mode=KgToRuleMode.FROM_EDGES)
''' Extract a Rule for each KG Edge'''
BinnedInfluences = RuleExtractionMethod(mode=KgToRuleMode.BINNED_INFLUENCES)
''' Extract Rules from KG by discretizing influence values and creating Rules from Influence-Parameter pairs'''
BinnedMergedInfluences = RuleExtractionMethod(
    mode=KgToRuleMode.BINNED_INFLUENCES_MERGED)
''' Extract Rules from KG by discretizing influence values and creating Rules from Influence-Parameter pairs, and then trying to merge rules with similar relative parameter changes'''

BinnedInfluencesFunction = partial(
    RuleExtractionMethod, mode=KgToRuleMode.BINNED_INFLUENCES_FUNCTION)
''' Extract Rules from KG by fitting a provided Estimator pipeline to the influence-parameter values and sampling this function subsequently'''



def merge_similar_rules(
    rules: Mapping[str, Iterable[Rule]],
    inf_key: str,
    parameter_df: pd.DataFrame,
    rel_parameter_values: Mapping[str,
                                  Mapping[str,
                                          MutableMapping[str,
                                                         Iterable[float]]]],
    parameter_values: Mapping[str, Mapping[str,
                                           MutableMapping[str,
                                                          Iterable[float]]]],
    label_encoder: LabelEncoderForColumns | None = None,
    _bins: str | int | Sequence[int | float]='doane'
) -> list[Rule]:
    """merge rules with equal / similar relative parameter changes

    Args:
        rules (Mapping[str, Iterable[Rule]]): dictionary of (condition range, list of Rules)
        inf_key (str): influence/condition name
        parameter_df (pd.DataFrame): KG parameter dataframe
        rel_parameter_values (Mapping[str, Mapping[str, MutableMapping[str, Iterable[float]]]]): nested dictionary of "raw" relative parameter values: dict[influence name][condition range][parameter name]
        parameter_values (Mapping[str, Mapping[str, MutableMapping[str, Iterable[float]]]]): nested dictionary of "raw" absolute parameter values: dict[influence name][condition range][parameter name]
        label_encoder (LabelEncoderForColumns | None): label encoder. Defaults to None.
        _bins (str | int | Sequence[int | float], optional) : see `np.histogram_bin_edges`. Defaults to 'doane'.

    Returns:
        list[Rule]: new list of rules, containing new merged and old unmergeable rules
    """

    def calculate_bin_edges_for_all_rules(
        all_rules: Mapping[str, Iterable[Rule]],
        bins: str | int | Sequence[int | float]='doane'
    ) -> dict[str, npt.NDArray[np.float64]]:
        """calculate `np.histogram_bin_edges` for all rules of all parameter types

        Args:
            all_rules (Mapping[str, Iterable[Rule]]): dictionary[condition range] => list[Rule]
            bins (str | int | Sequence[int | float], optional) : see `np.histogram_bin_edges`. Defaults to 'doane'.
        Returns:
            dict[str, npt.NDArray[np.float64]]: dictionary[parameter name] => np.array[bin edges]
        """

        def calculate_bin_edges(l_r: list[Rule], _bins: str | int | Sequence[int | float]):
            step_sizes = np.array([r.step_size for r in l_r])
            bins_ = np.histogram_bin_edges(step_sizes[~np.isnan(step_sizes)], bins=_bins)
            # TODO best binning function? 'auto' migh be better. should be passed as function argument explicitly in the future
            return bins_

        dict_bin_edges: defaultdict[str, npt.NDArray[np.float64]] = defaultdict()
        dict_rules: defaultdict[str, list[Rule]] = defaultdict(list)
        for influence_range in all_rules.keys():
            for parameter_key_ in parameter_df.keys():
                param_rules = [
                    r for r in all_rules[influence_range]
                    if r.parameter == parameter_key_
                ]
                dict_rules[parameter_key_].extend(param_rules)

        for parameter_key, rule_list in dict_rules.items():
            bin_edges = calculate_bin_edges(rule_list, bins)
            dict_bin_edges[parameter_key] = bin_edges
        return dict_bin_edges

    def rule_equality(rule_a: Rule, rule_b: Rule, bin_edges: npt.NDArray[np.float64]):
        """check if rules are similar enough to be merged.
        for string or boolean rules, if same action quantifier, for numerical if step_size is in the same bin
        """
        if not Rule.highlevel_eq(rule_a, rule_b):
            raise ValueError("differing condition or parameter")
        if not Rule.midlevel_eq(rule_a, rule_b):
            return False

        if rule_a.parameter_type in [Rule.ParamType.STRING, Rule.ParamType.BOOLEAN]:
            return rule_a.action_quantifier == rule_b.action_quantifier
        elif rule_a.parameter_type == Rule.ParamType.NUMERICAL:
            if rule_a.step_size is None or rule_a.step_size is None or np.isnan(
                    rule_a.step_size) or np.isnan(rule_b.step_size):
                raise ValueError("trying to compare empty bins")
            return custom_digitize(rule_a.step_size,
                                   bin_edges) == custom_digitize(rule_b.step_size, bin_edges)
        else:
            raise ValueError(f"unknown parameter type: {rule_a.parameter_type}")

    dict_bin_edges_by_parameter = calculate_bin_edges_for_all_rules(rules, _bins)

    output_rules: dict[str, list[Rule]] = defaultdict(list)
    for parameter_key in parameter_df.keys():
        rules_for_param = sorted([(bin_, r) for (bin_, l) in rules.items()
                                  for r in l if r.parameter == parameter_key],
                                 key=lambda t: t[1].condition_range)

        while rules_for_param:
            # greedy search for possible merge candidates
            # - pop "parent" from list
            # - try to find rule to merge:
            # either    - add new merged rule to input list and remove both "parents" from it
            # or        - add popped rule to output
            # repeat until input list is empty

            (candidate_a_bin_range, candidate_a_rule) = rules_for_param.pop(0)
            successful_merge_candidate = tuple()
            new_rule_entry: tuple[str, Rule] | None = None
            for (candidate_b_bin_range, candidate_b_rule) in rules_for_param:
                try:
                    rules_are_mergeable = rule_equality(
                        candidate_a_rule, candidate_b_rule,
                        dict_bin_edges_by_parameter[parameter_key])
                except ValueError as e:
                    logging.exception(e)
                    logging.error(
                        f"a: {candidate_a_rule},  b: {candidate_b_rule}")
                    continue
                if rules_are_mergeable:
                    logging.debug(
                        f"merge candiate found: {candidate_a_rule} - {candidate_b_rule}"
                    )
                    logging.debug(
                        f"bins: {dict_bin_edges_by_parameter[parameter_key]} indizes: A, B: {custom_digitize(candidate_a_rule.step_size, dict_bin_edges_by_parameter[parameter_key])}, {custom_digitize(candidate_b_rule.step_size, dict_bin_edges_by_parameter[parameter_key])}"
                    )

                    logging.debug("raw relative param values: ")
                    a_rel_param_vals = rel_parameter_values[inf_key][
                        candidate_a_bin_range][parameter_key]
                    b_rel_param_vals = rel_parameter_values[inf_key][
                        candidate_b_bin_range][parameter_key]
                    logging.debug(
                        f"relative A:  {a_rel_param_vals}\trelative B: {b_rel_param_vals}"
                    )
                    ab_rel_param_vals = a_rel_param_vals + b_rel_param_vals
                    logging.debug(
                        f"relative mean: {np.mean(ab_rel_param_vals)}\tstd: {np.std(ab_rel_param_vals)}"
                    )

                    logging.debug("raw absolute parameter values: ")
                    a_abs_param_vals = parameter_values[inf_key][
                        candidate_a_bin_range][parameter_key]
                    b_abs_param_vals = parameter_values[inf_key][
                        candidate_b_bin_range][parameter_key]
                    logging.debug(
                        f"absolute A:  {a_abs_param_vals}\tabsolute B: {b_abs_param_vals}"
                    )
                    logging.debug(
                        f"candidate A:  {a_abs_param_vals}\tcandidate B: {b_abs_param_vals}"
                    )
                    ab_abs_param_vals = a_abs_param_vals + b_abs_param_vals
                    logging.debug(
                        f"mean: {np.mean(ab_abs_param_vals)}\tstd: {np.std(ab_abs_param_vals)}"
                    )

                    new_condition_range = (candidate_a_rule.condition_range[0],
                                           candidate_b_rule.condition_range[1])
                    parameter_type = candidate_a_rule.parameter_type
                    r = Rule.from_relation(parameter_key, inf_key,
                                           parameter_type, ab_abs_param_vals,
                                           ab_rel_param_vals,
                                           new_condition_range, label_encoder)
                    logging.debug(str(r))
                    range_str = str(new_condition_range)
                    # update dictionary of "raw values" for consecutive merges
                    parameter_values[inf_key][range_str][
                        parameter_key] = ab_abs_param_vals
                    rel_parameter_values[inf_key][range_str][
                        parameter_key] = ab_rel_param_vals

                    # keep track of new rules and which original rules to remove later on
                    successful_merge_candidate = (candidate_b_bin_range,
                                                  candidate_b_rule)
                    new_rule_entry = (range_str, r)

                    # break loop, update iterables and start again
                    break

            if new_rule_entry is not None:
                # remove "parent" rules and add newly merged one
                rules_for_param.remove(successful_merge_candidate)
                rules_for_param.insert(0, new_rule_entry)
            else:
                # if no merge candidate was found, add the rule to final output
                output_rules[candidate_a_bin_range].append(candidate_a_rule)

    output = [rule for l in output_rules.values() for rule in l]

    return output


def custom_digitize(x: npt.ArrayLike,
                    bins: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """return indices of the bins to which values in input array belongs, with the rightmost bin including the right edge (x <= bins[i_max])
    """
    x = np.array(x)
    x = np.expand_dims(x, axis=0) if x.ndim == 0 else x
    indices = np.digitize(x, bins)
    indices_include_right_edge = np.digitize(x, bins, right=True)
    min_indices = np.amin([indices, indices_include_right_edge], axis=0)

    # find all rows which lay exactly on the right bin edge and put them in the bin below
    indices_right_edge = indices >= len(bins)
    indices[indices_right_edge] = min_indices[indices_right_edge]

    return indices


def df_filtered_by_influence_and_bin(param_df: pd.DataFrame, inf: str,
                                     inf_df: pd.DataFrame,
                                     bin_edges: npt.NDArray[np.float64],
                                     bin_index: int) -> pd.DataFrame:
    """Computes parameters by an influence by filtering a parameter dataframe for an influence which lies within the same given bin edges by only keeping common experiments (non equal None/Nan).

    Args:
        param_df (pd.DataFrame): dataframe containing parameters
        inf (str): name of the influence, has to be column name of inf_df
        inf_df (pd.DataFrame): dataframe containing quality influences
        bin_edges (npt.NDArray[np.float64]): array of bin edges (as created by `np.histogram_bin_edges`)
        bin_index (int): the index in which the influence value should be in

    Returns:
        pd.DataFrame: parameter dataframe with rows filtered to keep only those experiments where the influence appeared within the same value range
    """

    # get experiment_ids of influence with values inside bin
    bin_indices = (custom_digitize(inf_df[inf], bin_edges) == bin_index)
    experiments_with_inf = inf_df[inf].loc[bin_indices].dropna().index
    # use ids to select fittings rows
    return param_df.loc[experiments_with_inf]


def calculate_bins(
        dataframes_dict: Mapping[str, pd.DataFrame],
        binning_function: str | int | Sequence[int | float] = 'doane',
        relative_influences=False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """calculate bin edges for the influence/condition columns in the dataframe

    Args:
        dataframes_dict (Mapping[str, pd.DataFrame]): dictionary with keys 'influence', 'influences_relative' containing dataframes of influence values
        binning_function (str | int | Sequence[int | float], optional): binning function, see `np.histogram_bin_edges`. Defaults to 'doane'.
        relative_influences (bool, optional): if absolute or relative influence dataframe should be read from dataframes_dict. Defaults to False.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: Dictionary of influence bin data: dictionary[influence name] = tuple(histogram, bin_edges)
    """

    influences_df = dataframes_dict[
        'influences_relative'] if relative_influences else dataframes_dict[
            'influences']

    bins = {}
    for influence_key in influences_df:
        values = np.array(influences_df[influence_key].dropna())
        hist, bin_edges = np.histogram(values, bins=binning_function)

        bins[influence_key] = (hist, bin_edges)

    return bins


def df_filtered_by_influence(param_df: pd.DataFrame, inf: str,
                             inf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes parameters by an influence by filtering a parameter dataframe for an influence by only keeping common experiments (non-equal None/Nan).

    Args:
        param_df (pd.DataFrame): dataframe containing parameters
        inf (str): name of the influence, has to be column name of inf_df
        inf_df (pd.DataFrame): dataframe containing quality influences

    Returns:
        pd.DataFrame: parameter dataframe with rows filtered to keep only those experiments where the influence appeared
    """

    # get experiment_ids of influence with values inside bin
    experiments_with_inf = inf_df[inf].dropna().index
    # use ids to select fittings rows
    return param_df.loc[experiments_with_inf]


def kg_to_improvement_rules(
        knowledge_graph: igraph.Graph,
        experiment_series: dict,
        boolean_parameters: Iterable[str],
        string_parameters: Iterable[str],
        rule_extraction_method: RuleExtractionMethod = FromEdge,
        **kwargs) -> list[Rule]:
    """
    A knowledge graph is converted to improvement rules. Without Binning and merging, that means every edge \
    that has a quality influence as source vertex will be converted to a rule.
    Otherwise, the changed (relative) parameters and influences dataframes created from the KG will be used instead.

    Args:
        knowledge_graph (igraph.Graph): Knowledge Graph instance
        experiment_series (dict): experiment series data
        boolean_parameters (Iterable[str]: list of parameters of type boolean
        string_parameters (Iterable[str]): list of parameters of type string
        rule_extraction_method (RuleExtractionMethod): method on how to extract the rules from KG.
        use predefined ones in this package. supply pipeline when using `BinnedInfluencesFunction`. Defaults to FromEdge.

    Returns:
        list[Rule]: list of created Rule objects from knowledge graph
    """
    all_dfs = kg.prepare_dfs_of_experiments(knowledge_graph,
                                            experiment_series,
                                            fill_nones=False)
    relative_params_df = all_dfs['changed_parameters_relative']
    params_df = all_dfs['changed_parameters']
    influences_df = all_dfs['influences']

    param_values, rel_param_values = {}, {}

    influence_bin_data = calculate_bins(all_dfs)
    # remove environment influences
    influence_bin_data: dict[str, tuple[np.ndarray, np.ndarray]] = {
        key: influence_bin_data[key]
        for key in all_dfs['influences_relative'].columns
    }
    mode = rule_extraction_method.mode
    label_encoder = kwargs.pop('label_encoder', None)


    if mode == KgToRuleMode.FROM_EDGES:

        def get_start_and_target_vertices(edge_):
            return edge_.source_vertex, edge_.target_vertex

        list_of_expert_rules = list()

        for edge in knowledge_graph.es:
            influence, parameter = get_start_and_target_vertices(edge)
            # leave out rules that only have an environment condition
            # TODO XAI-558 does this make sense?
            if influence['type'] != 'qual_influence':
                continue

            if parameter['name'] in boolean_parameters:
                param_type = Rule.ParamType.BOOLEAN
            elif parameter['name'] in string_parameters:
                param_type = Rule.ParamType.STRING
            else:
                param_type = Rule.ParamType.NUMERICAL

            rule = rule_from_edge(relative_params_df, params_df, influences_df,
                                  influence_bin_data, influence['name'],
                                  parameter['name'], param_type, label_encoder, **kwargs)

            list_of_expert_rules.append(rule)

        return list_of_expert_rules

    elif mode == KgToRuleMode.BINNED_INFLUENCES_FUNCTION:
        kwargs['method'] = rule_extraction_method
        returned_rules, _, _, _ = create_sampled_rules(
            boolean_parameters, string_parameters, relative_params_df,
            params_df, influences_df, influence_bin_data, label_encoder, **kwargs)
    else:
        returned_rules, param_values, rel_param_values = create_binned_rules(
            boolean_parameters, string_parameters, relative_params_df,
            params_df, influences_df, influence_bin_data, label_encoder)

    list_of_expert_rules = []
    # optionally merge created rules on relative parameter values
    if mode == KgToRuleMode.BINNED_INFLUENCES_MERGED:
        for inf in influence_bin_data.keys():
            merged_rules = merge_similar_rules(returned_rules[inf], inf,
                                               params_df, rel_param_values,
                                               param_values, label_encoder)
            list_of_expert_rules.extend(merged_rules)

    else:
        # returned_rules: defaultdict[str, defaultdict[str, list[Rule]]]
        list_of_expert_rules = [
            r for nested in returned_rules.values() for l in nested.values()
            for r in l
        ]

    return list_of_expert_rules


def rule_from_edge(
    relative_params_df: pd.DataFrame,
    params_df: pd.DataFrame,
    influences_df: pd.DataFrame,
    influence_bin_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    influence_name: str,
    parameter_name: str,
    param_type: Rule.ParamType = Rule.ParamType.UNKNOWN,
    label_encoder: LabelEncoderForColumns | None = None,
    **kwargs
) -> Rule:
    """create a Rule from a Knowledge Graph relation (edge).

    Args:
        relative_params_df (pd.DataFrame): dataframe containing changed relative parameter values
        params_df (pd.DataFrame): dataframe containing changed absolute parameter values
        influences_df (pd.DataFrame): dataframe containing influence (condition) values
        influence_bin_data (Mapping[str, tuple[np.ndarray, np.ndarray]]): dictionary[influence name] = result of `np.histogram`= (histogram, bin_edges)
        influence_name (str): name of the influence
        parameter_name (str): name of the parameter
        param_type (Rule.ParamType, optional): Rule parameter type. Defaults to Rule.ParamType.UNKNOWN.

    Returns:
        Rule: rule created from KG edge
    """
    if label_encoder is None:
        label_encoder = data_provider_singleton.get_label_encoder(**kwargs)


    _, condition_bin_edges = influence_bin_data[influence_name]
    x = influences_df[influence_name].mean()
    bin_index = np.squeeze(custom_digitize(x, condition_bin_edges))
    try:
        lower_bin_edge = condition_bin_edges[bin_index - 1]
        upper_bin_edge = condition_bin_edges[bin_index]
    except IndexError as error:
        logging.error(error)
        logging.error(
            f"x: {x}, index: {bin_index}, bins: {condition_bin_edges}")
        std = influences_df[influence_name].std()
        logging.info(f"using +- std instead: {std}")
        lower_bin_edge = x - std
        upper_bin_edge = x + std

    condition_range = (lower_bin_edge, upper_bin_edge)

    influenced_parameters = df_filtered_by_influence(params_df, influence_name,
                                                     influences_df)
    influenced_relative_parameters = df_filtered_by_influence(
        relative_params_df, influence_name, influences_df)

    param_values = influenced_parameters[parameter_name].dropna()
    try:
        relative_params_values = influenced_relative_parameters[
            parameter_name].dropna()
    except KeyError:
        relative_params_values = None
    rule = Rule.from_relation(parameter_name, influence_name, param_type,
                              param_values, relative_params_values,
                              condition_range, label_encoder)
    return rule


def find_best_classifier(X: npt.ArrayLike, y: npt.ArrayLike):
    """ fit SVC classifier to data. """

    no_classes = len(np.unique(y.ravel()))
    if no_classes < 2:
        classifier = sklearn.dummy.DummyClassifier(strategy='most_frequent')
    else:
        classifier = sklearn.svm.SVC(
        )  # just use the most basic Support Vector Classifier with default parameters for now
    classifier.fit(X, y)
    # TODO: GridSearchCV for classification
    return classifier


def find_best_regressor(X: npt.ArrayLike,
                        y: npt.ArrayLike,
                        pipeline: Pipeline | None = None,
                        parameters: list[dict] | None = None,
                        scoring: str | Callable = "r2",
                        cv=3):
    """run GridSearchCV to find optimal parameters using cross-validation. default pipeline and arguments are used if not supplied

    Args:
        X (npt.ArrayLike): training data.
        y (npt.ArrayLike): target values.
        pipeline (sklearn.pipeline.Pipeline, optional): regression pipeline, including optional preprocessing. Defaults to None.
        parameters (list[dict] | None, optional): list of dicts for different Pipelines (i.e. different Regressors) and their parameter ranges. Defaults to None.
        scoring (str | Callable): metric to use for scoring. Defaults to "r2"

    Returns:
        sklearn.pipeline.Pipeline: pipeline
    """
    no_samples = np.count_nonzero(y)
    if no_samples < 2:
        regressor = sklearn.dummy.DummyRegressor(strategy="mean")
        regressor.fit(X, y)
        return regressor
    elif cv == False and pipeline is not None:
        # if no CV is wanted, just fit the setup pipeline on the whole dataset
        pipeline.fit(X, y)
        return pipeline
    else:
        regressor = lm.LinearRegression()
        regressor.fit(X, y)
        return regressor


# not used right now, may be dropped into `find_best_regressor` function instead of just using LinearRegression for everything
def run_grid_search(X: npt.ArrayLike, y: npt.ArrayLike,
                    pipeline: Pipeline | None, parameters: list[dict] | None,
                    scoring: str | Callable, cv: int | Iterable | Any):
    """ run cross-validated grid-search over a parameter grid to find the best regressor for the data.

    Args:
        X (npt.ArrayLike): training vector
        y (npt.ArrayLike): target relative to X
        pipeline (Pipeline | None): preprocessing pipeline with estimator as last step
        parameters (list[dict] | None): list of parameter setup grid with lists or tuples of possible values
        scoring (str | Callable): performance evaluation strategy, e.g. 'f1', 'r2', 'neg_mean_squared_error', etc.
        cv (int | Iterable | Any): cross-validation splitting strategy, number of folds, sklearn CV Splitter or an (train, test) iterable

    Returns:
        Estimator: best fitted GridSearch-Estimator
    """
    no_samples = np.count_nonzero(y)
    min_samples = (2 * cv) if isinstance(cv, int) else 10
    if no_samples < min_samples:
        if no_samples < 2:
            logging.warning(
                "less than 2 samples. using mean DummyRegressor instead!")
            regressor = sklearn.dummy.DummyRegressor(strategy="mean")
            regressor.fit(X, y)
            return regressor
        else:
            logging.info(
                "not enough samples. using basic LinearRegression instead.")
            regressor = lm.LinearRegression()
            regressor.fit(X, y)
            return regressor
    if pipeline is None:
        pipeline = Pipeline([
            ('standardize', prep.StandardScaler()),
            ('poly', prep.PolynomialFeatures()),
            ('regressor', None),
        ])
    if parameters is None:
        poly_degrees = (1, 2, 3)
        parameters = [
            {
                'poly__degree': poly_degrees,
                'regressor': (lm.LinearRegression(), ),
            },
            {
                'poly__degree': poly_degrees,
                'regressor': (sklearn.svm.SVR(), ),
                'regressor__kernel': ['linear', 'rbf'],
                'regressor__C': (0.01, 1.0, 10),
                'regressor__gamma': (0.01, 1.0, 10),
                'regressor__max_iter': (10000, ),
            },
            {
                'poly__degree': poly_degrees,
                'regressor': (lm.LassoLarsCV(normalize=False), ),
            },
            # {
            #     'poly__degree': poly_degrees,
            #     'regressor': (lm.SGDRegressor(),),
            #     'regressor__max_iter': (10000,),
            # },
            {
                'poly__degree': poly_degrees,
                'regressor': (lm.RidgeCV(), ),
            },
        ]
    grid_search = GridSearchCV(pipeline,
                               parameters,
                               scoring=scoring,
                               error_score="raise",
                               cv=cv,
                               n_jobs=-1,
                               verbose=0)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            grid_search.fit(X, y)
        except (UndefinedMetricWarning, ValueError):
            logging.warning(
                "not enough samples. using mean DummyRegressor instead!")
            regressor = sklearn.dummy.DummyRegressor(strategy="mean")
            regressor.fit(X, y)
            return regressor
    return grid_search


def create_sampled_rules(
        boolean_parameters: Iterable[str],
        string_parameters: Iterable[str],
        relative_params_df: pd.DataFrame,
        params_df: pd.DataFrame,
        influences_df: pd.DataFrame,
        influence_bin_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
        label_encoder: LabelEncoderForColumns | None = None,
        **rule_fun_kwargs):
    """ Create Rules from dataframes by sampling a function describing the parameter-influence relationships.

    First, the functions are estimated by using a LinearRegression/Classifier or cross-validated GridSearch fitted specifically to each p-q-pair. These functions are then sampled at the discrete bins defined in `influence_bin_data` to generate rules.

    Args:
        boolean_parameters (Iterable[str]): list of parameter names of boolean type
        string_parameters (Iterable[str]): list of parameter names of string type
        relative_params_df (pd.DataFrame): dataframe containing changed relative parameter values
        params_df (pd.DataFrame): dataframe containing changed absolute parameter values
        influences_df (pd.DataFrame): dataframe containing influence values
        influence_bin_data (Mapping[str, tuple[np.ndarray, np.ndarray]]): dictionary[influence name] = result of `np.histogram`= (histogram, bin_edges)
        label_encoder (LabelEncoderForColumns | None): label encoder to transform string/boolean columns. Defaults to None.
        rule_fun_kwargs (dict): set cross validation splits at 'cv'. set `RuleExtractionMethod` at key 'method'

    Returns:
        tuple of four dictionaries: created sampled rules, fitted estimator objects for relative and absolute parameter values, as well as the original datatables used to fit the estimators.

        sampled rules[influence name][condition/influence bin range]=list[Rule]

        fitted estimators relative[influence name][parameter name] = sklearn Estimator (Classifier / Regressor)

        fitted estimators absolute[influence name][parameter name]= sklearn Estimator (Classifier / Regressor)

        estimation data[influence name][parameter name]= data table (dataframe) which was used for fitting the estimators, with columns (influence name, parameter name)
    """

    def create_influence_parameter_table(inf_df, param_df, inf_key, param_key):
        """create two-column table for use in linear regression"""
        y_ = df_filtered_by_influence(param_df, inf_key, inf_df)
        y_ = y_[param_key].dropna().round(2)
        x = inf_df[inf_key].loc[y_.index]
        xy = pd.merge(x, y_, left_index=True, right_index=True)
        xy.sort_values(inf_key, inplace=True)
        return xy

    sampled_rules: defaultdict[str, defaultdict[
        str, list[Rule]]] = defaultdict(lambda: defaultdict(list))

    estimation_data = defaultdict(lambda: defaultdict())
    fitted_estimators = defaultdict(lambda: defaultdict())
    fitted_estimators_absolute = defaultdict(lambda: defaultdict())

    method: RuleExtractionMethod = rule_fun_kwargs['method']

    cross_val = rule_fun_kwargs.get('cv', False)
    # for each influence-parameter pair, run a grid search to find best fitted regression

    for influence in influences_df.columns:
        for parameter in relative_params_df.columns:
            # created two-column table (idx, x, y) for both relative and absolute parameter values
            table = create_influence_parameter_table(influences_df,
                                                     relative_params_df,
                                                     influence, parameter)
            table_abs = create_influence_parameter_table(
                influences_df, params_df, influence, parameter)

            if table.empty:
                continue

            X = table[influence].values.reshape((-1, 1))
            y = table[parameter].values

            X_abs = table_abs[influence].values.reshape((-1, 1))
            y_abs = table_abs[parameter]

            logging.debug(
                f"starting grid search for : {influence}-{parameter}-pair")

            # try to fit classifiers for boolean and categorical, regressors for numerical tables, that best describe the p-q-relationship
            if parameter in [*boolean_parameters, *string_parameters]:
                estimator = find_best_classifier(X, y)
                estimator_abs = find_best_classifier(X_abs, y_abs)
            else:
                pipeline = method.make_new_pipeline()
                pipeline_abs = method.make_new_pipeline()
                estimator = find_best_regressor(X, y, pipeline=pipeline, cv=cross_val)
                estimator_abs = find_best_regressor(X_abs, y_abs, pipeline=pipeline_abs, cv=cross_val)

            estimator_total_score = estimator.score(X, y)
            if isinstance(estimator, GridSearchCV):
                logging.debug(
                    f"best CV score: {estimator.best_score_} with {estimator.scoring} for regressor {estimator.best_estimator_}"
                )
            logging.debug(
                f"fitted regressor to {influence}-{parameter}-pair, score: {estimator_total_score}"
            )

            estimation_data[influence][parameter] = table
            fitted_estimators[influence][parameter] = estimator

            if isinstance(estimator_abs, GridSearchCV):
                logging.debug(
                    f"best CV score (abs): {estimator_abs.best_score_}")
            fitted_estimators_absolute[influence][parameter] = estimator_abs

    # sample function at discrete (baseline) influence values to create rules
    # look through all (quality) influences and their value bins
    for influence_key, (_, bin_edges) in influence_bin_data.items():
        # iterate pairwise over the tuples of bin edges
        # (might aswell use a resampling function directly)
        for lower, upper in pairwise(bin_edges):
            mean_bin_val = np.mean([lower, upper])
            bin_idx = custom_digitize(np.mean([lower, upper]), bin_edges)

            lower = np.round(lower, 1)
            upper = np.round(upper, 1)

            # select parameters affected by the specific influence in the specific bin interval
            influenced_parameters_df = df_filtered_by_influence_and_bin(
                params_df, influence_key, influences_df, bin_edges, bin_idx)

            for parameter_key in params_df.keys():
                # only sample the function in this bin and create a rule, if there are any parameter changes for this influence-value
                param_values = influenced_parameters_df[parameter_key].dropna()
                if param_values.empty:
                    continue

                try:
                    # sample the function in this area
                    estimator = fitted_estimators[influence_key][parameter_key]
                    estimator_abs = fitted_estimators_absolute[influence_key][parameter_key]
                    y_pred = estimator.predict(mean_bin_val.reshape(-1, 1))
                    y_pred_abs = estimator_abs.predict(
                        mean_bin_val.reshape(-1, 1))

                except KeyError:
                    continue

                rule = create_rule(boolean_parameters, string_parameters,
                                   parameter_key, influence_key, y_pred_abs,
                                   y_pred, (lower, upper), label_encoder)

                sampled_rules[influence_key][str((lower, upper))].append(rule)

    return sampled_rules, fitted_estimators, fitted_estimators_absolute, estimation_data


def create_binned_rules(
    boolean_parameters: Iterable[str],
    string_parameters: Iterable[str],
    relative_params_df: pd.DataFrame,
    params_df: pd.DataFrame,
    influences_df: pd.DataFrame,
    influence_bin_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    label_encoder: LabelEncoderForColumns | None = None
) -> tuple[defaultdict[str, defaultdict[str, list[Rule]]], defaultdict[
        str, defaultdict[str, defaultdict[str, list[float]]]], defaultdict[
            str, defaultdict[str, defaultdict[str, list[float]]]]]:
    """create rules from KG for each influenced parameter, by discretizing influence values into bins in which rules will be created

    Args:
        boolean_parameters (Iterable[str]): list of parameter names of boolean type
        string_parameters (Iterable[str]): list of parameter names of string type
        relative_params_df (pd.DataFrame): dataframe containing changed relative parameter values
        params_df (pd.DataFrame): dataframe containing changed absolute parameter values
        influences_df (pd.DataFrame): dataframe containing influence values
        influence_bin_data (Mapping[str, tuple[np.ndarray, np.ndarray]]): dictionary[influence name] = result of `np.histogram`= (histogram, bin_edges)
        label_encoder (LabelEncoderForColumns | None): label encoder to transform string/boolean columns. Defaults to None.

    Returns:
        tuple of three dictionaries: created binned rules, absolute and relative "raw" parameter values on which the averaged rule parameters are based on.

        binned rules[influence name][condition/influence bin range]=list[Rule]

        parameter values[influence name][condition/influence bin range][parameter name] = list[raw parameter values for merging]

        relative parameter values[influence name][condition/influence bin range][parameter name]= list[raw relative parameter values for merging]
    """

    binned_rules: defaultdict[str, defaultdict[str, list[Rule]]] = defaultdict(
        lambda: defaultdict(list))
    parameter_values: defaultdict[str, defaultdict[str, defaultdict[
        str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
    relative_parameter_values: defaultdict[str, defaultdict[str, defaultdict[
        str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))

    # look through all (quality) influences and their value bin edges
    for influence_key, (_, bin_edges) in influence_bin_data.items():
        # iterate pairwise over the tuples of bin edges
        for lower, upper in pairwise(bin_edges):
            bin_idx = custom_digitize(np.mean([lower, upper]), bin_edges)
            # select parameters affected by the specific influence in the specific bin interval
            influenced_parameters_df = df_filtered_by_influence_and_bin(
                params_df, influence_key, influences_df, bin_edges, bin_idx)
            influenced_relative_parameters_df = df_filtered_by_influence_and_bin(
                relative_params_df, influence_key, influences_df, bin_edges,
                bin_idx)
            # create rules for all (non-empty) parameters for the current influence in the current value range (bin)
            for parameter_key in params_df.keys():
                param_values = influenced_parameters_df[parameter_key].dropna()

                lower = np.round(lower, 1)
                upper = np.round(upper, 1)

                # make sure there are entries in the dict, even if parameters are "nan"
                if binned_rules[influence_key][str((lower, upper))] is None:
                    binned_rules[influence_key][str((lower, upper))] = []
                if parameter_values[influence_key][str(
                    (lower, upper))] is None:
                    parameter_values[influence_key][str(
                        (lower, upper))][parameter_key] = []
                if relative_parameter_values[influence_key][str(
                    (lower, upper))][parameter_key] is None:
                    relative_parameter_values[influence_key][str(
                        (lower, upper))][parameter_key] = []
                if param_values.empty:
                    continue
                try:
                    relative_param_values = influenced_relative_parameters_df[
                        parameter_key].dropna()
                except KeyError:
                    relative_param_values = None
                # if influence['type'] != 'qual_influence':
                #     continue
                rule = create_rule(boolean_parameters, string_parameters,
                                   parameter_key, influence_key, param_values,
                                   relative_param_values, (lower, upper),
                                   label_encoder)

                binned_rules[influence_key][str((lower, upper))].append(rule)
                # memorize the original parameter values used to create the averaged rules, for merging rules later on
                parameter_values[influence_key][str(
                    (lower, upper))][parameter_key].extend(param_values)
                if relative_param_values is not None:
                    relative_parameter_values[influence_key][str(
                        (lower,
                         upper))][parameter_key].extend(relative_param_values)

    return binned_rules, parameter_values, relative_parameter_values


def create_rule(boolean_parameters: Iterable[str],
                string_parameters: Iterable[str],
                parameter_name: str,
                influence_name: str,
                influenced_parameters: pd.Series,
                influenced_relative_parameters: pd.Series | None,
                influence_bin: tuple[float, float],
                label_encoder: LabelEncoderForColumns | None = None) -> Rule:
    """create a Rule Object from the given KG relation data."""
    if label_encoder is None:
        label_encoder = data_provider_singleton.get_label_encoder()

    if parameter_name in boolean_parameters:
        param_type = Rule.ParamType.BOOLEAN
    elif parameter_name in string_parameters:
        param_type = Rule.ParamType.STRING
    else:
        param_type = Rule.ParamType.NUMERICAL
    rule = Rule.from_relation(parameter_name, influence_name, param_type,
                              influenced_parameters,
                              influenced_relative_parameters, influence_bin,
                              label_encoder)

    return rule
