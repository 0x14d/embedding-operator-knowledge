from __future__ import annotations
from functools import reduce

import logging
import os
from collections import defaultdict
from typing import List, Sequence, Callable, Iterable, Tuple, Type, Union
import igraph
import pandas as pd

from data_provider.synthetic_data_provider import SyntheticDataProvider
from preprocessing import LabelEncoderForColumns
from rule_base.rule import Rule
from rule_base.quantified_conditions_rule \
    import QuantifiedConditionsRule
from rule_base.quantified_conclusions_rule \
    import QuantifiedConclusionsRule
from rule_base.rule_extraction import RuleExtractionMethod, BinnedMergedInfluences, kg_to_improvement_rules
from rule_base.rule_serializer import RuleSerializer
from utilities import Path


def compare_rules(rule_set_one, rule_set_two, list_of_levels):
    """
    compare two rule sets
    It will be checked on three different levels if the rules in rule set two are approved by the rules in set 1
    High level: relation between influence and parameter is approved
    Mid level: action to be done for the parameter is approved
    low level: the same range of values or the same action quantifier is chosen
    :param rule_set_one first set of rules
    :param rule_set_two second set of rules
    :param list_of_levels list of levels for which the rule sets should be compared
    """
    approved_rules = dict()
    for level in list_of_levels:
        approved_rules[level.name] = list()

    for rule_one in rule_set_one:
        for rule_two in rule_set_two:
            for level in list_of_levels:
                if level(rule_one, rule_two) is True:
                    approved_rules[level.name].append(rule_one)
    return approved_rules


def evaluate_rules(ground_truth: Sequence[Rule],
                   predicted_rules: Sequence[Rule],
                   equality_function: Callable[[Rule, Rule], bool], metrics):
    """
    Given an equality_function which defines the level where rules should be compared several metrics are evaluated for the given rules
    :param ground_truth:
    :param predicted_rules:
    :param equality_function:
    :param metrics --> members of ExperimentDefinition.MetricIdentifier
    :return:
    """
    stats = dict()

    sub_ground_truth_rules, sub_pred_rules, _ = prepare_evaluation_sets(
        ground_truth, predicted_rules, equality_function)
    for metric in metrics:
        stats[metric.name] = metric(list(sub_ground_truth_rules.values()),
                                    list(sub_pred_rules.values()),
                                    len(predicted_rules))
    return stats


def prepare_evaluation_sets(ground_truth: Iterable[Rule],
                            prediction: Iterable[Rule],
                            equality_function: Callable[[Rule, Rule], bool]):
    """
    Substitutes rules that are approx. equal in baseline & pred with a common rule and converts them returns a dictionary that encodes whether a rule was detected or not
    :param ground_truth:
    :param prediction:
    :param equality_function:
    :return: ( baselines_as_1dbool, prediction_as_1dbool , lookup of replaced rules)
    """
    # substitute equal rules
    unique_ground_truth = ground_truth.copy()
    unique_prediction = prediction.copy()
    shared_rules = []
    substituted_rules = {}
    for ground_truth_rule in ground_truth.copy():
        for prediction_rule in prediction.copy():
            if equality_function(ground_truth_rule, prediction_rule):
                # we substitute baseline rules with prediction rules bc they are the ones we are interested in later
                # check  whether we already detected this rule though!
                if not any([str(prediction_rule) == str(shared) for shared in shared_rules]):
                    shared_rules.append(prediction_rule)
                    substituted_rules[str(prediction_rule)] = ground_truth_rule
                # remove it
                unique_ground_truth = [gt for gt in unique_ground_truth if str(gt) != str(ground_truth_rule)]
                unique_prediction = [pred for pred in unique_prediction if str(pred) != str(prediction_rule)]

    # initialize the sets: 1 if is present in all rules 0 otherwise
    ground_truth_as_1dbool = {
        **{rule: 1 for rule in shared_rules},
        **{rule: 1 for rule in unique_ground_truth},
        **{rule: 0 for rule in unique_prediction}
    }
    prediction_as_1dbool = {
        **{rule: 1 for rule in shared_rules},
        **{rule: 0 for rule in unique_ground_truth},
        **{rule: 1 for rule in unique_prediction}
    }
    assert (len(ground_truth_as_1dbool) == len(prediction_as_1dbool))
    # if we have shared rules both sets should be 1 for this rule
    if len(shared_rules) > 0:
        assert (ground_truth_as_1dbool[shared_rules[0]] == 1
                and prediction_as_1dbool[shared_rules[0]] == 1)

    return ground_truth_as_1dbool, prediction_as_1dbool, substituted_rules


def write_kg_rules_to_excel(
        knowledge_graph: igraph.Graph,
        experiment_series,
        boolean_parameters: Iterable[str],
        string_parameters: Iterable[str],
        path: Path = "./survey.xlsx",
        method: RuleExtractionMethod = BinnedMergedInfluences,
        **kwargs):
    """
    writes all rules that can be found in the given knowledge graph to an excel sheet for a expert survey
    :param knowledge_graph knowledge graph from which the rules should be extracted
    :param experiment_series dict with experiment series so that the relative improvements can be calculated
    :return no return
    """
    logging.info("writing KG rules to excel...")
    all_rules = kg_to_improvement_rules(knowledge_graph,
                                        experiment_series,
                                        boolean_parameters,
                                        string_parameters,
                                        rule_extraction_method=method,
                                        **kwargs)
    all_rules.sort(
        key=lambda x: (x.condition, x.parameter, x.condition_range[0]))
    columns = [
        'Rule Candidate', "Rule Corrected", "Correct Relation",
        "Correct Action", "Correct Quantifier"
    ]
    rule_exporter = RuleSerializer()
    rule_exporter.serialize(
        rules=all_rules,
        output_format=RuleSerializer.Format.EXCEL,
        version=RuleSerializer.Version.QUANTIFIED_INFLUENCES,
        path=path,
        header=columns,
        **kwargs)


def load_expert_survey_baseline_rules(dirname: Path,
                                      label_encoder: LabelEncoderForColumns,
                                      threshold=0.6,
                                      version=RuleSerializer.Version.VERSION3,
                                      **kwargs):
    """
    evaluates the expert surveys and tells you how many of the rules are accepted by the experts -- used for
    establishing a baseline of the knowledge graph
    # TODO XAI-558 provide statistics?
    :param dirname relative path to the directory where all expert surveys are stored
    :param threshold desired percentage of experts that have to accept a certain rule
    :return an unfiltered list containing all rules selected by experts; a dictionary in which the string of a rule is
    the key and the value is the percentage of how many experts accepted this rule filtered by threshold;
    """

    def rule_shorthand(rule: Rule):
        return f'{rule.parameter}-{rule.condition}'

    stats = {}
    file_names = [
        os.path.join(dirname, f) for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f))
    ]
    stats['number of experts'] = len(file_names)
    expert_corrected_rules = dict()
    merge_candidates = {}
    # import the accepted rules from the expert surveys
    rt = RuleSerializer()
    cols = ['Rule Candidate', 'Rule Corrected']
    for survey in file_names:
        rules_dict: dict[str, list[Rule]] = rt.deserialize(
            input_format=RuleSerializer.Format.EXCEL,
            label_encoder=label_encoder,
            path=survey,
            version=version,
            usecols=cols)  # type: ignore
        expert_corrected_rules[survey] = rules_dict['Rule Corrected']
    stats['amount rules per expert'] = [
        len(rules) for expert, rules in expert_corrected_rules.items()
    ]
    # create dict with list of rules to be able to aggregate them
    for survey, rules in expert_corrected_rules.items():
        if not merge_candidates:
            merge_candidates = {rule_shorthand(rule): [rule] for rule in rules}
        else:
            for rule in rules:
                try:
                    merge_candidates[rule_shorthand(rule)].append(rule)
                except KeyError:
                    merge_candidates[rule_shorthand(rule)] = [rule]
    stats['total_amount_of_rules'] = [
        len(rules) for _, rules in merge_candidates.items() if len(rules)
    ]
    stats['agreements_on_rules'] = [
        len(rules) for _, rules in merge_candidates.items() if len(rules) >= 1
    ]
    stats['agreements_on_rules_one_expert'] = [
        len(rules) for _, rules in merge_candidates.items() if len(rules) >= 1
    ]
    stats['agreements_on_rules_two_experts'] = [
        len(rules) for _, rules in merge_candidates.items() if len(rules) >= 2
    ]
    stats['agreements_on_rules_three_experts'] = [
        len(rules) for _, rules in merge_candidates.items() if len(rules) >= 3
    ]
    merged_rules: list[Rule] = []
    for shorthand, rules in merge_candidates.items():
        # check whether we have a high enough degree of agreement between the experts
        if len(rules) / len(file_names) < threshold:
            continue
        if version == RuleSerializer.Version.KCAP:
            merged_rule = QuantifiedConclusionsRule.merge_rules(
                rules, label_encoder)
        elif version == RuleSerializer.Version.VERSION3:
            merged_rule = QuantifiedConditionsRule.merge_rules(
                rules, label_encoder)
        else:
            raise NotImplementedError(
                f'Merging of version {version} rules isn\'t implemented')
        merged_rules.append(merged_rule)

    return merged_rules, stats


def load_synthetic_groundtruth_rules(
    data_provider: SyntheticDataProvider,
    label_encoder: LabelEncoderForColumns,
    rule_class: Union[Type[QuantifiedConditionsRule], Type[QuantifiedConclusionsRule]] 
) -> Tuple[List[Rule], None]:
    """Loads the groundtruth rules from the synthetic data generator."""
    relations = data_provider.data_generator.pq_relations
    rules = reduce(
        lambda rules, rel: rules + rel.to_rules(label_encoder, rule_class),
        relations,
        []
    )
    return rules, None


def write_results_to_excel(groundtruth_name: str,
                           results: dict,
                           method_substitutions=None,
                           base_directory: str = "./"):
    """
    creates a csv file that contains all the evaluated weight and filter methods
    You will see how many of the given rules (e.g. simplify3D) were approved in the graph at high/mid/low level
    :param results dictionary with the data to be written in the csv file
    """

    # reorder results to be able to use pandas multiindex
    reordered_dict = defaultdict(dict)
    for rule_extraction_method in results.keys():
        for level in results[rule_extraction_method].keys():
            for method in results[rule_extraction_method][level].keys():
                if method_substitutions:
                    if method in method_substitutions:
                        subs_method = method_substitutions[method]
                        for metric in results[rule_extraction_method][level][
                                method].keys():
                            reordered_dict[(rule_extraction_method, level,
                                            subs_method)].update({
                                                metric:
                                                results[rule_extraction_method]
                                                [level][method][metric]
                                            })
                else:
                    for metric in results[rule_extraction_method][level][
                            method].keys():
                        reordered_dict[(rule_extraction_method, level,
                                        method)].update({
                                            metric:
                                            results[rule_extraction_method]
                                            [level][method][metric]
                                        })
    df = pd.DataFrame.from_dict(reordered_dict, orient='index')
    os.makedirs(base_directory, exist_ok=True)
    df.to_excel(f'{base_directory}overall_statistics_{groundtruth_name.lower()}.xls')
    df.to_latex(f'{base_directory}overall_statistics_{groundtruth_name.lower()}.tex',
                float_format=lambda x: '%10.2f' % x,
                escape=False)


def write_rules_to_file(fname: str,
                        ruleset: Iterable[Rule],
                        version=RuleSerializer.Version.UNKNOWN,
                        base_directory: str = "./"):
    base_directory = f'{base_directory}rules'
    os.makedirs(base_directory, exist_ok=True)
    with open(f'{base_directory}/{fname}.txt', 'w') as f:
        rt = RuleSerializer()
        rules = rt.serialize(ruleset,
                             output_format=RuleSerializer.Format.STRING,
                             version=version)
        if rules is not None:
            for rule in sorted(rules):
                f.write("%s\n" % str(rule))
