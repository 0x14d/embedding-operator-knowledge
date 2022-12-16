import os

import pytest

from kbc_evaluation.dataset import DataSet
from kbc_evaluation.evaluator import EvaluationRunner, Evaluator


class TestEvaluator:
    def test_evaluator_failure(self):
        with pytest.raises(Exception):
            EvaluationRunner(file_to_be_evaluated="xzy")
        with pytest.raises(Exception):
            EvaluationRunner()

    def test_hits_at(self):
        test_file_path = "./tests/test_resources/eval_test_file.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18
        )
        assert runner.calculate_hits_at(1)[2] == 2
        assert runner.calculate_hits_at(3)[2] == 3
        assert runner.calculate_hits_at(10)[2] == 4

    def test_hits_at_with_confidence(self):
        test_file_path = "./tests/test_resources/eval_test_file_with_confidences.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18
        )
        assert runner.calculate_hits_at(1)[2] == 2
        assert runner.calculate_hits_at(3)[2] == 3
        assert runner.calculate_hits_at(10)[2] == 4

    def test_calculate_results_no_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_with_confidences.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        results = Evaluator.calculate_results(
            file_to_be_evaluated=test_file_path,
            data_set=DataSet.WN18,
            n=1,
        )
        assert results.filtered_hits_at_n_all == 2
        assert results.filtered_hits_at_n_all >= results.filtered_hits_at_n_all
        assert results.n == 1

        # simple type assertions
        assert type(results.evaluated_file) == str
        assert type(results.n) == int
        assert type(results.filtered_hits_at_n_heads) == int
        assert type(results.filtered_hits_at_n_tails) == int
        assert type(results.filtered_hits_at_n_all) == int

        results = Evaluator.calculate_results(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18, n=3
        )
        assert results.filtered_hits_at_n_all == 3
        assert results.filtered_hits_at_n_all >= results.filtered_hits_at_n_all
        assert results.n == 3

        results = Evaluator.calculate_results(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18, n=10
        )
        assert results.filtered_hits_at_n_all == 4
        assert results.filtered_hits_at_n_all >= results.filtered_hits_at_n_all
        assert results.n == 10

        test_file_path = "./tests/test_resources/eval_test_file.txt"
        assert os.path.isfile(test_file_path)
        results = Evaluator.calculate_results(
            # n and data_set are irrelevant here
            file_to_be_evaluated=test_file_path,
            data_set=DataSet.WN18,
            n=4,
        )
        assert results.non_filtered_mean_rank_all == 3

    def test_hits_at_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_filtering.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path,
            is_apply_filtering=True,
            data_set=DataSet.WN18,
        )

        # [2] encodes "all"
        assert runner.calculate_hits_at(1)[2] == 3
        assert runner.calculate_hits_at(3)[2] == 9

        # [0]: heads, [1]: tails, [2]: all
        assert runner.calculate_hits_at(10)[0] == 6
        assert runner.calculate_hits_at(10)[1] == 5
        assert runner.calculate_hits_at(10)[2] == 11

    def test_calculate_results_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_filtering.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        results = Evaluator.calculate_results(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18, n=1
        )
        assert results.filtered_hits_at_n_all == 3  # n is 1
        assert results.n == 1
        assert type(results.filtered_hits_at_n_all) == int  # n is 1
        assert type(results.filtered_hits_at_n_tails) == int
        assert type(results.filtered_hits_at_n_heads) == int

    def test_hits_at_filtering_with_confidence(self):
        test_file_path = (
            "./tests/test_resources/eval_test_file_filtering_with_confidences.txt"
        )
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path,
            data_set=DataSet.WN18,
            is_apply_filtering=True,
        )
        assert runner.calculate_hits_at(1)[2] == 2
        assert runner.calculate_hits_at(3)[2] == 4
        assert runner.calculate_hits_at(10)[2] == 6

    def test_mean_rank(self):
        test_file_path = "./tests/test_resources/eval_test_file.txt"
        assert os.path.isfile(test_file_path)
        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18
        )

        ranks = runner.mean_rank()

        # [0] encodes mean rank heads
        assert round((6 + 1) / 2) == ranks[0]

        # [1] encodes mean rank tails
        assert round((3 + 1) / 2) == ranks[1]

        # [2] encodes mean rank all
        assert 3 == ranks[2]

        # [3] encodes mrr heads
        assert (1 / 6 + 1) / 2 == ranks[3]

        # [4] encodes mrr tails
        assert (1 / 3 + 1) / 2 == ranks[4]

        # [5] encodes mrr all
        mrr_all = (1 / 6 + 1 + 1 / 3 + 1) / 4
        assert mrr_all == ranks[5]
        
        # [6] encodes amri head
        amri_head = 1 - (((6 + 1)/2-1)/(5.5-1))
        assert amri_head == ranks[6]
        
        # [7] encodes amri tail
        amri_tail = 1 - (((3 + 1)/2-1)/(5.5 - 1))
        assert amri_tail == ranks[7]

        # [8] encodes amri both side
        amri_all = (amri_head + amri_tail) / 2 
        assert amri_all == ranks[8]
        
        

    def test_mean_rank_with_confidence(self):
        test_file_path = "./tests/test_resources/eval_test_file_with_confidences.txt"
        assert os.path.isfile(test_file_path)
        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18
        )
        rank = runner.mean_rank()
        assert 3 == rank[2]
        assert (1 / 6 + 1 + 1 / 3 + 1) / 4 == rank[5]

    def test_mean_rank_with_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_filtering.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path,
            is_apply_filtering=True,
            data_set=DataSet.WN18,
        )
        result = runner.mean_rank()

        # note that the result is rounded
        # as explained in eval_test_file_filtering_readme.md, MR for H is 3.333 and for T is 2.6
        # rounded, this is 3
        # the weighted (!) average is exactly three
        assert 3 == result[0]  # MR head
        assert 3 == result[1]  # MR tail
        assert 3 == result[2]  # MR all

        # test MRR
        mrr_heads = (1 / 6 + 1 / 6 + 1 + 1 / 3 + 1 / 3 + 1) / 6
        assert mrr_heads == result[3]

        mrr_tails = (1 / 3 + 1 / 3 + 1 + 1 / 3 + 1 / 3) / 5
        assert mrr_tails == result[4]

        mrr = (6 * mrr_heads + 5 * mrr_tails) / 11
        assert mrr == result[5]

    def test_write_results_to_file(self):
        test_file_path = "./tests/test_resources/eval_test_file.txt"
        assert os.path.isfile(test_file_path)
        Evaluator.write_results_to_file(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18
        )
        assert os.path.isfile("./results.txt")
        os.remove("./results.txt")
        Evaluator.write_results_to_file(
            file_to_be_evaluated=test_file_path,
            file_to_be_written="./results_test.txt",
            data_set=DataSet.WN18,
        )
        assert os.path.isfile("./results_test.txt")
        os.remove("./results_test.txt")

    def test_filtering_with_fb15k(self):
        test_file_path = "./tests/test_resources/freebase_filtering_example.txt"
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path,
            data_set=DataSet.FB15K,
            is_apply_filtering=True,
        )
        hits_at_2 = runner.calculate_hits_at(2)
        assert hits_at_2[0] == 1
        assert hits_at_2[1] == 1
        assert hits_at_2[2] == 2

        hits_at_1 = runner.calculate_hits_at(1)
        assert hits_at_1[0] == 0
        assert hits_at_1[1] == 0
        assert hits_at_1[2] == 0
