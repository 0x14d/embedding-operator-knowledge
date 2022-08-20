import logging
import pickle
import unittest
import numpy as np
import os
import sklearn.preprocessing
from sklearn import linear_model as lm
import pandas as pd
from .rule_extraction import BinnedInfluencesFunction, create_sampled_rules, merge_similar_rules, create_binned_rules

from .rule import Rule


class InfluenceQuantificationTests(unittest.TestCase):

    def setUp(self):
        import os
        if os.path.basename(os.getcwd()) == 'rule_base':
            os.chdir("..")

        with open('./obj/label_encoder.pkl', 'rb') as pkl:
            self.label_encoder = pickle.load(pkl)

        self.boolean_parameters = ['cool_fan_enabled']
        self.string_parameters = ['adhesion_type']

        data = {
            'param_1':
            np.concatenate((np.random.rand(10) * 10, np.repeat(np.nan, 6))),
            'adhesion_type':
            np.concatenate((
                np.repeat(np.nan, 10),
                np.array([0.0, 0.0, 2.0]),  # "brim","brim","raft"
                np.repeat(np.nan, 3))),
            'cool_fan_enabled':
            np.concatenate(
                (np.repeat(np.nan,
                           13), np.array([1, 1,
                                          0]))),  # "True", "True", "False"
        }

        self.abs_params_df = pd.DataFrame(data)

        self.rel_params_df = pd.DataFrame.from_dict({
            'param_1':
            np.concatenate((np.array([10, 11, 12, 13, 14, 15, 50, np.nan,
                                      51]), np.repeat(np.nan, 7))),
        })

        self.influences_df = pd.DataFrame.from_dict({
            "influence_a":
            [1, 1.09, 0.96, 1.15, 3, 6, -1, np.nan, 0, 1] +  # numerical \
            [0.99, 1.5, 2.3] +  # string \
            [1.0, 1.6, 2.2],  # boolean
        })

        self.influence_bins = {
            "influence_a":
            (np.array([1, 1, 3, 1, 0, 1, 0, 0, 0, 1]),
             np.array([-1., -0.3, 0.4, 1.1, 1.8, 2.5, 3.2, 3.9, 4.6, 5.3,
                       6.])),
        }

        self.binned_rules_str = [
            "If you encounter influence_a in [-1.0, -0.3], try to increase incrementally the parameter param_1 by 50.0",
            "If you encounter influence_a in [-0.3, 0.4], try to increase incrementally the parameter param_1 by 51.0",
            "If you encounter influence_a in [0.4, 1.1], try to increase incrementally the parameter param_1 by 11.0",
            "If you encounter influence_a in [1.1, 1.8], try to increase incrementally the parameter param_1 by 13.0",
            "If you encounter influence_a in [2.5, 3.2], try to increase incrementally the parameter param_1 by 14.0",
            "If you encounter influence_a in [5.3, 6.0], try to increase incrementally the parameter param_1 by 15.0",
            "If you encounter influence_a in [0.4, 1.1], try to set the parameter adhesion_type to brim",
            "If you encounter influence_a in [1.1, 1.8], try to set the parameter adhesion_type to brim",
            "If you encounter influence_a in [1.8, 2.5], try to set the parameter adhesion_type to raft",
            "If you encounter influence_a in [0.4, 1.1], try to set the parameter cool_fan_enabled to True",
            "If you encounter influence_a in [1.1, 1.8], try to set the parameter cool_fan_enabled to True",
            "If you encounter influence_a in [1.8, 2.5], try to set the parameter cool_fan_enabled to False",
        ]

        self.binned_rules = [
            Rule.from_string(r, self.label_encoder)
            for r in self.binned_rules_str
        ]

        str_binned_merged_rules = [
            "If you encounter influence_a in [-1.0, 0.4], try to increase incrementally the parameter param_1 by 50.5",
            "If you encounter influence_a in [0.4, 3.2], try to increase incrementally the parameter param_1 by 12.0",
            "If you encounter influence_a in [5.3, 6.0], try to increase incrementally the parameter param_1 by 15.0",
            "If you encounter influence_a in [0.4, 1.8], try to set the parameter adhesion_type to brim",
            "If you encounter influence_a in [1.8, 2.5], try to set the parameter adhesion_type to raft",
            "If you encounter influence_a in [0.4, 1.8], try to set the parameter cool_fan_enabled to True",
            "If you encounter influence_a in [1.8, 2.5], try to set the parameter cool_fan_enabled to False",
        ]
        self.binned_merged_rules_str = str_binned_merged_rules
        self.binned_merged_rules = [
            Rule.from_string(r, self.label_encoder)
            for r in self.binned_merged_rules_str
        ]


        self.sampled_rules_str = [
            "If you encounter influence_a in [0.0, 0.9], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [0.9, 1.8], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [1.8, 2.7], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [2.7, 3.6], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [3.6, 4.5], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [4.5, 5.4], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [5.4, 6.3], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [6.3, 7.2], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [7.2, 8.1], try to increase incrementally the parameter param_1 by 2.0",
            "If you encounter influence_a in [8.1, 9.0], try to increase incrementally the parameter param_1 by 2.0",
        ]

        self.sampled_rules = [Rule.from_string(r, self.label_encoder) for r in self.sampled_rules_str]
        

    @staticmethod
    def get_sampled_reference_data() -> tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame,dict]:
        # return influences, absolute and relative params dataframes, influence bins
        influence = np.arange(0,10)
        influence_bins = np.histogram(influence)
        # f(q) = 2x+10
        abs_params = 2*influence+10
        rel_params = np.diff(abs_params, prepend=2*-1+10)

        influence_df = pd.DataFrame.from_dict({"influence_a": influence})
        abs_params_df = pd.DataFrame.from_dict({"param_1": abs_params})
        rel_params_df = pd.DataFrame.from_dict({"param_1": rel_params})
        influence_bins_dict = {"influence_a": influence_bins}
        return influence_df, abs_params_df, rel_params_df, influence_bins_dict

    @property
    def merged_rules(self):
        binned_rules_set = set(self.binned_rules)
        merged_rules_set = set(self.binned_merged_rules)
        diff = merged_rules_set.difference(binned_rules_set)
        return diff

    def test_rule_binning(self):
        binned_rules, _, _ = create_binned_rules(
            self.boolean_parameters,
            self.string_parameters,
            self.rel_params_df,
            self.abs_params_df,
            self.influences_df,
            self.influence_bins,
            label_encoder=self.label_encoder)
        key = 'influence_a'

        reference = set(self.binned_rules_str)

        result = set()
        for k in binned_rules[key].keys():
            result.update([str(r) for r in binned_rules[key][k]])
        binned_as_expected = result.intersection(reference)
        if len(binned_as_expected) < 1 < len(result):
            logging.warning(
                f"no intersection! ret: {result}, ref: {reference}")
        else:
            logging.info(f"found expected: {binned_as_expected}")
        logging.info(f"ref not found: {reference.difference(result)}")
        logging.info(f"binned not in ref: {result.difference(reference)}")

        self.assertEqual(result, reference)

    def test_rule_merging(self):
        binned_rules, param_values, rel_param_values = create_binned_rules(
            self.boolean_parameters,
            self.string_parameters,
            self.rel_params_df,
            self.abs_params_df,
            self.influences_df,
            self.influence_bins,
            label_encoder=self.label_encoder)
        merged_rules = merge_similar_rules(binned_rules['influence_a'],
                                           'influence_a',
                                           self.abs_params_df,
                                           rel_param_values,
                                           param_values,
                                           label_encoder=self.label_encoder)

        reference = set(self.binned_merged_rules_str)

        result = set([str(r) for r in merged_rules])
        merged_as_expected = result.intersection(reference)
        if len(merged_as_expected) < 1 < len(result):
            logging.warning(
                f"no intersection! ret: {result}, ref: {reference}")
        else:
            logging.info(f"found expected: {merged_as_expected}")
        logging.info(f"ref not found: {reference.difference(result)}")
        logging.info(f"merged not in ref: {result.difference(reference)}")

        self.assertEqual(result, reference)


    def test_rule_sampling(self):
        influences_df, abs_params_df, rel_params_df, influence_bins = self.get_sampled_reference_data()
        pipeline = {"LINEAR": [
                ('standardize', sklearn.preprocessing.StandardScaler()),
                ('poly', sklearn.preprocessing.PolynomialFeatures(1)),
                ('regressor', lm.LinearRegression())],}
        pipe_name = "LINEAR"
        method = BinnedInfluencesFunction(pipeline_steps=pipeline[pipe_name], pipeline_name=pipe_name)
        sampled_rules, fitted_estimators, fitted_estimators_absolute, estimation_data = create_sampled_rules(self.boolean_parameters,
                                                 self.string_parameters,
                                                 rel_params_df,
                                                 abs_params_df,
                                                 influences_df,
                                                 influence_bins,
                                                 method=method,
                                                 label_encoder=self.label_encoder)
        key = 'influence_a'

        reference = set(self.sampled_rules_str)

        result = set()
        for k in sampled_rules[key].keys():
            result.update([str(r) for r in sampled_rules[key][k]])
        sampled_as_expected = result.intersection(reference)
        if len(sampled_as_expected) < 1 and len(result) > 1:
            logging.warning(
                f"no intersection! ret: {result}, ref: {reference}")
        else:
            logging.info(f"found expected: {sampled_as_expected}")
        logging.info(f"ref not found: {reference.difference(result)}")
        logging.info(f"sampled not in ref: {result.difference(reference)}")

        self.assertEqual(result, reference)