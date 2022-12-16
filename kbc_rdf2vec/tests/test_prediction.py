import numpy as np
from gensim.models import KeyedVectors

from kbc_rdf2vec.dataset import DataSet
from kbc_rdf2vec.prediction import (
    PredictionFunctionEnum,
    RandomPredictionFunction,
    AveragePredicateAdditionPredictionFunction,
)


class TestPredictionFunction:
    def test_get_instance(self) -> None:
        """A simple test checking that get_instance always returns an implementation for all enum values."""
        kv = KeyedVectors.load("./tests/test_resources/mini_3d_wn_model.kv", mmap="r")
        for function in PredictionFunctionEnum:
            assert (
                function.get_instance(keyed_vectors=kv, data_set=DataSet.WN18)
                is not None
            )

    def test_implementation_of_all_prediction_functions(self) -> None:
        """Test whether implementations exist, whether reflexivity is implemented, and whether the correct result is
        contained in the result list."""
        kv = KeyedVectors.load("./tests/test_resources/mini_3d_wn_model.kv", mmap="r")
        for function in PredictionFunctionEnum:
            # forbid reflexive
            function_instance = function.get_instance(
                keyed_vectors=kv,
                data_set=DataSet.WN18,
                is_reflexive_match_allowed=False,
            )
            assert function_instance is not None

            h = "09590495"
            l = "_synset_domain_topic_of"
            t = "09689152"

            result_h_prediction = function_instance.predict_heads([h, l, t], n=None)
            # make sure that the tail is not predicted when predicting heads
            assert t not in result_h_prediction

            # make sure that the correct h appears in the prediction
            assert h in [x[0] for x in result_h_prediction]

            # make sure that the returned list is sorted in descending order of the confidence
            smallest_confidence = 100.0
            for p, confidence in result_h_prediction:
                assert (
                    smallest_confidence >= confidence
                ), f"Result not correctly sorted for {function}"
                if confidence < smallest_confidence:
                    smallest_confidence = confidence

            # perform test on tail prediction
            result_t_prediction = function_instance.predict_tails([h, l, t], n=None)

            # make sure that the head is not predicted when predicting tails
            assert (
                h not in result_t_prediction
            ), f"Failure for {function}: Found head as prediction of tail."

            # make sure that the solution appears for tail predictions
            assert t in [
                x[0] for x in result_t_prediction
            ], f"Failure for {function}: Did not find correct prediction of tail."

            smallest_confidence = 100.0
            for p, confidence in result_t_prediction:
                assert (
                    smallest_confidence >= confidence
                ), f"Result not correctly sorted for {function}"
                if confidence < smallest_confidence:
                    smallest_confidence = confidence

            # allow reflexive (but exclude most similar due to implementation of gensim excluding those always)
            if (
                function == PredictionFunctionEnum.MOST_SIMILAR
                or function == PredictionFunctionEnum.PREDICATE_AVERAGING_MOST_SIMILAR
            ):
                continue

            function_instance = function.get_instance(
                keyed_vectors=kv, data_set=DataSet.WN18, is_reflexive_match_allowed=True
            )
            assert function_instance is not None

            # make sure the solution is found (head prediction)
            result = function_instance.predict_heads([h, l, t], n=None)
            assert h in (
                item[0] for item in result
            ), f"Head {h} not found in prediction of function {function}."

            # make sure the solution is found (tail prediction)
            result = function_instance.predict_tails([h, l, t], n=None)
            assert t in (
                item[0] for item in result
            ), f"Tail {t} not found in prediction of function {function}."


class TestRandomPredictionFunction:
    def test_predict_heads(self):
        """Tests the N parameter at various positions for head predictions. We cannot check for the solution here
        because we use a text kv model."""
        kv = KeyedVectors.load("./tests/test_resources/wn_test_model.kv", mmap="r")
        rpf = RandomPredictionFunction(
            keyed_vectors=kv, data_set=DataSet.WN18, is_reflexive_match_allowed=False
        )

        # the test data
        h = "09590495"
        l = "_synset_domain_topic_of"
        t = "09689152"

        # n = 10
        result = rpf.predict_heads([h, l, t], n=10)
        assert len(result) == 10

        # n = 18
        result = rpf.predict_heads([h, "_synset_domain_topic_of", "09689152"], n=18)
        assert len(result) == 18

        # n = None
        result = rpf.predict_heads([h, "_synset_domain_topic_of", "09689152"], n=None)
        assert len(result) > 100

    def test_predict_tails(self):
        """Tests the N parameter at various positions for tail predictions. We cannot check for the solution here
        because we use a text kv model."""
        kv = KeyedVectors.load("./tests/test_resources/wn_test_model.kv", mmap="r")
        rpf = RandomPredictionFunction(keyed_vectors=kv, data_set=DataSet.WN18)
        result = rpf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=10
        )
        assert len(result) == 10
        result = rpf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=23
        )
        assert len(result) == 23
        result = rpf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=None
        )
        assert len(result) > 100


class TestAveragePredicatePredictionFunction:
    def test_constructor(self):
        kv = KeyedVectors.load("./tests/test_resources/mini_3d_wn_model.kv", mmap="r")
        appf = AveragePredicateAdditionPredictionFunction(
            keyed_vectors=kv, data_set=DataSet.WN18
        )
        # just checking whether the constructor works:
        assert appf is not None

        # test tail prediction n = 15
        result = appf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=15
        )
        assert len(result) == 15

        # test tail prediction n = all
        result = appf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=None
        )
        assert len(result) > 100

        result = appf.predict_heads(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=15
        )
        assert len(result) == 15

        # test head prediction n = all
        result = appf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=None
        )
        assert len(result) > 100

    def test_axioms_np(self):
        """Simple axioms to ensure correct usage."""
        my_list = [np.array([1, 2, 3]), np.array([3, 4, 5]), np.array([5, 6, 7])]
        mean_vector = np.mean(my_list, axis=0)
        assert mean_vector[0] == 3
        assert mean_vector[1] == 4
        assert mean_vector[2] == 5
