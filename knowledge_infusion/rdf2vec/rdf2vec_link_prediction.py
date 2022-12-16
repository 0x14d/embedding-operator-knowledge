""" This module provides an adapter from the aipe pipeline to the KBC Evaluation
framework. 
"""

from kbc_rdf2vec.prediction import PredictionFunctionEnum
from kbc_rdf2vec.rdf2vec_kbc import Rdf2vecKbc
from kbc_evaluation.evaluator import Evaluator
from typing import Dict
import os
import pickle


def calculate_amri_hits_at_k_for_rdf2vec(base_folder: str) -> Dict:
    """Calculates the AMRI and Hit@K metric for an existing embedding created
    with the rdf2vec method

    Args:
        base_folder (str): Base folder of the embedding.

    Returns:
        Dict: results of the calculation
    """

    # Open some files generatted during the embedding
    nt_file = os.path.join(base_folder, "dataset.nt")
    model = os.path.join(base_folder, "model.model")

    with open(base_folder + "eval_dataset_object.pickle", "rb") as in_f:
        dataset = pickle.load(in_f)

    # Generate the prediction making class, make predictions and save them to the
    # FS
    kbc = Rdf2vecKbc(
        model_path=model,
        n=None,
        data_set=dataset,
        file_for_predicate_exclusion=nt_file,
        is_reflexive_match_allowed=False,
        # This function was choosen cause it is in the spirit of 10.3233/SW-212892
        prediction_function=PredictionFunctionEnum.PREDICATE_AVERAGING_ADDITION,
    )
    kbc.predict(os.path.join(base_folder, "predictions.txt"))

    # Generate the hits@k metrik for different ks
    # The AMRI is also generated in each of those Evaluators, since the AMRI
    # does not have a parameter k, it has the same value in each of the 3 result
    # objects.
    results_at_1 = Evaluator.calculate_results(
        file_to_be_evaluated=os.path.join(base_folder, "predictions.txt"),
        data_set=dataset,
        n=1,
    )
    results_at_5 = Evaluator.calculate_results(
        file_to_be_evaluated=os.path.join(base_folder, "predictions.txt"),
        data_set=dataset,
        n=5,
    )
    results_at_10 = Evaluator.calculate_results(
        file_to_be_evaluated=os.path.join(base_folder, "predictions.txt"),
        data_set=dataset,
        n=10,
    )

    # Put the results into a dictionary
    metric_results = {}
    # head
    metric_results[
        ("adjustedarithmeticmeanrankindex", "head")
    ] = results_at_10.filtered_amri_heads
    metric_results[("hits_at_1", "head")] = results_at_1.filtered_hits_at_n_heads
    metric_results[("hits_at_5", "head")] = results_at_5.filtered_hits_at_n_heads
    metric_results[("hits_at_10", "head")] = results_at_10.filtered_hits_at_n_heads
    # tail
    metric_results[
        ("adjustedarithmeticmeanrankindex", "tail")
    ] = results_at_10.filtered_amri_tails
    metric_results[("hits_at_1", "tail")] = results_at_1.filtered_hits_at_n_tails
    metric_results[("hits_at_5", "tail")] = results_at_5.filtered_hits_at_n_tails
    metric_results[("hits_at_10", "tail")] = results_at_10.filtered_hits_at_n_tails

    return metric_results
