import logging.config
import os
from typing import Tuple

from kbc_evaluation.dataset import DataSet, ParsedSet

logconf_file = os.path.join(os.path.dirname(__file__), "log.conf")
logging.config.fileConfig(fname=logconf_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)



class EvaluatorResult:
    """Object holding the results of the evaluation process"""

    def __init__(
        self,
        evaluated_file: str,
        test_set_size: int,
        n: int,
        filtered_hits_at_n_heads: int,
        filtered_hits_at_n_tails: int,
        filtered_hits_at_n_all: int,
        filtered_mean_rank_heads: int,
        filtered_mean_rank_tails: int,
        filtered_mean_rank_all: int,
        filtered_reciprocal_mean_rank_heads: float,
        filtered_reciprocal_mean_rank_tails: float,
        filtered_reciprocal_mean_rank_all: float,
        filtered_amri_heads: float,
        filtered_amri_tails: float,
        filtered_amri_all: float,
        non_filtered_hits_at_n_heads: int,
        non_filtered_hits_at_n_tails: int,
        non_filtered_hits_at_n_all: int,
        non_filtered_mean_rank_heads: int,
        non_filtered_mean_rank_tails: int,
        non_filtered_mean_rank_all: int,
        non_filtered_reciprocal_mean_rank_heads: float,
        non_filtered_reciprocal_mean_rank_tails: float,
        non_filtered_reciprocal_mean_rank_all: float,
        non_filtered_amri_heads: float,
        non_filtered_amri_tails: float,
        non_filtered_amri_all: float,
    ):
        # setting the general variables
        self.evaluated_file = evaluated_file
        self.test_set_size = test_set_size
        self.n = n

        # setting the filtered results
        self.filtered_hits_at_n_heads = filtered_hits_at_n_heads
        self.filtered_hits_at_n_tails = filtered_hits_at_n_tails
        self.filtered_hits_at_n_all = filtered_hits_at_n_all
        self.filtered_hits_at_n_relative = self.filtered_hits_at_n_all / (
            2 * test_set_size
        )
        self.filtered_mean_rank_heads = filtered_mean_rank_heads
        self.filtered_mean_rank_tails = filtered_mean_rank_tails
        self.filtered_mean_rank_all = filtered_mean_rank_all
        self.filtered_reciprocal_mean_rank_heads = filtered_reciprocal_mean_rank_heads
        self.filtered_reciprocal_mean_rank_tails = filtered_reciprocal_mean_rank_tails
        self.filtered_reciprocal_mean_rank_all = filtered_reciprocal_mean_rank_all

        # setting the non-filtered results
        self.non_filtered_hits_at_n_heads = non_filtered_hits_at_n_heads
        self.non_filtered_hits_at_n_tails = non_filtered_hits_at_n_tails
        self.non_filtered_hits_at_n_all = non_filtered_hits_at_n_all
        self.non_filtered_hits_at_n_relative = self.non_filtered_hits_at_n_all / (
            2 * test_set_size
        )
        self.non_filtered_mean_rank_heads = non_filtered_mean_rank_heads
        self.non_filtered_mean_rank_tails = non_filtered_mean_rank_tails
        self.non_filtered_mean_rank_all = non_filtered_mean_rank_all
        self.non_filtered_reciprocal_mean_rank_heads = (
            non_filtered_reciprocal_mean_rank_heads
        )
        self.non_filtered_reciprocal_mean_rank_tails = (
            non_filtered_reciprocal_mean_rank_tails
        )
        self.non_filtered_reciprocal_mean_rank_all = (
            non_filtered_reciprocal_mean_rank_all
        )
        
        self.filtered_amri_heads = filtered_amri_heads
        self.filtered_amri_tails = filtered_amri_tails
        self.filtered_amri_all = filtered_amri_all
        
        self.non_filtered_amri_heads = non_filtered_amri_heads
        self.non_filtered_amri_tails = non_filtered_amri_tails
        self.non_filtered_amri_all = non_filtered_amri_all


class EvaluationRunner:
    """This class calculates evaluation scores for a single file."""

    def __init__(
        self,
        file_to_be_evaluated: str,
        data_set: DataSet,
        is_apply_filtering: bool = False,
    ):
        """Constructor

        Parameters
        ----------
        file_to_be_evaluated : str
            Path to the text file with the predicted links that shall be evaluated.
        data_set : DataSet
            The dataset for which predictions have been made.
        is_apply_filtering : bool
            Indicates whether filtering is desired (if True, results will likely improve).
        """

        self._file_to_be_evaluated = file_to_be_evaluated
        self._is_apply_filtering = is_apply_filtering

        if file_to_be_evaluated is None or not os.path.isfile(file_to_be_evaluated):
            logging.error(
                f"The evaluator will not work because the specified file "
                f"does not exist {file_to_be_evaluated}"
            )
            raise Exception(
                f"The specified file ({file_to_be_evaluated}) does not exist."
            )

        self.parsed = ParsedSet(
            is_apply_filtering=self._is_apply_filtering,
            file_to_be_evaluated=self._file_to_be_evaluated,
            data_set=data_set,
        )

    def mean_rank(self) -> Tuple[int, int, int, float, float, float]:
        """Calculates the mean rank and mean reciprocal rank using the given file.

        Returns
        -------
        Tuple[int, int, int, float, float, float]
        The first three elements are for MR, the last three for MRR.

        [0] Mean rank as int for heads (rounded float).
        [1] Mean rank as int for tails (rounded float).
        [2] Mean rank as int (rounded float).
        [3] Mean reciprocal rank as float for heads.
        [4] Mean reciprocal as float for tails.
        [5] Mean reciprocal as float.

        """
        ignored_heads = 0
        ignored_tails = 0
        head_rank = 0
        tail_rank = 0
        reciprocal_head_rank = 0
        reciprocal_tail_rank = 0
        
        n_head = 0
        n_tail = 0
        expected_head_rank = 0
        expected_tail_rank = 0

        for truth, prediction in self.parsed.triple_predictions.items():
            try:
                h_index = (
                    prediction[0].index(truth[0]) + 1
                )  # (first position has index 0)
                head_rank += h_index
                reciprocal_head_rank += 1.0 / h_index
                expected_head_rank += 0.5 * (len(prediction[0]) + 1)
                n_head += 1
            except ValueError:
                logging.error(
                    f"ERROR: Failed to retrieve head predictions for (correct) head concept: {truth[0]} "
                    f"Triple: {truth}"
                )
                ignored_heads += 1
            try:
                t_index = (
                    prediction[1].index(truth[2]) + 1
                )  # (first position has index 0)
                tail_rank += t_index
                reciprocal_tail_rank += 1.0 / t_index
                expected_tail_rank += 0.5 * (len(prediction[1]) + 1)
                n_tail += 1
            except ValueError:
                logging.error(
                    f"ERROR: Failed to retrieve tail predictions for (correct) tail concept: {truth[2]} "
                    f"Triple: {truth}"
                )
                ignored_tails += 1

        mean_head_rank = 0
        mean_tail_rank = 0
        mean_reciprocal_head_rank = 0
        mean_reciprocal_tail_rank = 0
        total_tasks = self.parsed.total_prediction_tasks
        if total_tasks - ignored_heads > 0:
            denominator = total_tasks / 2.0 - ignored_heads
            mean_head_rank = head_rank / denominator
            mean_reciprocal_head_rank = reciprocal_head_rank / denominator
        if total_tasks / 2 - ignored_tails > 0:
            denominator = total_tasks / 2 - ignored_tails
            mean_tail_rank = tail_rank / denominator
            mean_reciprocal_tail_rank = reciprocal_tail_rank / denominator


        # AMRI HEAD
        expected_head_rank /= n_head
        amri_head = 1 - ((mean_head_rank - 1) / (expected_head_rank - 1))
        
        # AMRI TAIL
        expected_tail_rank /= n_tail
        amri_tail = 1 - ((mean_tail_rank - 1) / (expected_tail_rank - 1))

        mean_rank = 0
        mean_reciprocal_rank = 0
        if (total_tasks - ignored_tails - ignored_heads) > 0:
            single_tasks = total_tasks / 2
            mean_rank = (
                head_rank / (single_tasks - ignored_heads)
                + tail_rank / (single_tasks - ignored_tails)
            ) / 2

            total_completed_tasks = total_tasks - ignored_tails - ignored_heads
            mean_reciprocal_rank = (
                mean_reciprocal_head_rank
                * ((single_tasks - ignored_heads) / total_completed_tasks)
                + mean_reciprocal_tail_rank
                * (single_tasks - ignored_tails)
                / total_completed_tasks
            )
        mean_rank_rounded = round(mean_rank)
        
        # AMRI BOTH
        amri = (amri_head + amri_tail) / 2
        
        return (
            round(mean_head_rank),
            round(mean_tail_rank),
            mean_rank_rounded,
            mean_reciprocal_head_rank,
            mean_reciprocal_tail_rank,
            mean_reciprocal_rank,
            amri_head,
            amri_tail,
            amri
        )

    def calculate_hits_at(self, n: int = 10) -> Tuple[int, int, int]:
        """Calculation of hits@n.

        Parameters
        ----------
        n : int
            Hits@n. This parameter specifies the n.

        Returns
        -------
        Tuple[int, int, int]
            [0] Hits at n only for heads.
            [1] Hits at n only for tails.
            [2] The hits at n. Note that head hits and tail hits are added.
        """
        heads_hits = 0
        tails_hits = 0

        for truth, prediction in self.parsed.triple_predictions.items():
            # perform the actual evaluation
            if truth[0] in prediction[0][:n]:
                heads_hits += 1
            if truth[2] in prediction[1][:n]:
                tails_hits += 1
        
        heads_hits /= len(self.parsed.triple_predictions)
        tails_hits /= len(self.parsed.triple_predictions)

        result = (heads_hits + tails_hits) / 2
        return heads_hits, tails_hits, result


class Evaluator:
    """This class provides powerful evaluation reporting capabilities."""

    @staticmethod
    def calculate_results(
        file_to_be_evaluated: str,
        data_set: DataSet,
        n: int = 10,
    ) -> EvaluatorResult:
        """Given the file_to_be_evaluated and a data_set, this method calculates hits at n.

        Parameters
        ----------
        file_to_be_evaluated : str
        data_set : DataSet
        n : int
            Hits@n. This parameter specifies the n. Default value 10.

        Returns
        -------
        EvaluatorResult
            The result data structure.
        """

        evaluator = EvaluationRunner(
            file_to_be_evaluated=file_to_be_evaluated,
            is_apply_filtering=False,
            data_set=data_set,
        )

        non_filtered_hits_at_10 = evaluator.calculate_hits_at(n)
        test_set_size = len(data_set.test_set())
        non_filtered_mr = evaluator.mean_rank()

        evaluator = EvaluationRunner(
            file_to_be_evaluated=file_to_be_evaluated,
            is_apply_filtering=True,
            data_set=data_set,
        )
        filtered_hits_at_10 = evaluator.calculate_hits_at(n)
        filtered_mr = evaluator.mean_rank()

        return EvaluatorResult(
            evaluated_file=file_to_be_evaluated,
            test_set_size=test_set_size,
            n=n,
            filtered_hits_at_n_heads=filtered_hits_at_10[0],
            filtered_hits_at_n_tails=filtered_hits_at_10[1],
            filtered_hits_at_n_all=filtered_hits_at_10[2],
            filtered_mean_rank_heads=filtered_mr[0],
            filtered_mean_rank_tails=filtered_mr[1],
            filtered_mean_rank_all=filtered_mr[2],
            filtered_reciprocal_mean_rank_heads=filtered_mr[3],
            filtered_reciprocal_mean_rank_tails=filtered_mr[4],
            filtered_reciprocal_mean_rank_all=filtered_mr[5],
            filtered_amri_heads=filtered_mr[6],
            filtered_amri_tails=filtered_mr[7],
            filtered_amri_all=filtered_mr[8],
            non_filtered_hits_at_n_heads=non_filtered_hits_at_10[0],
            non_filtered_hits_at_n_tails=non_filtered_hits_at_10[1],
            non_filtered_hits_at_n_all=non_filtered_hits_at_10[2],
            non_filtered_mean_rank_heads=non_filtered_mr[0],
            non_filtered_mean_rank_tails=non_filtered_mr[1],
            non_filtered_mean_rank_all=non_filtered_mr[2],
            non_filtered_reciprocal_mean_rank_heads=non_filtered_mr[3],
            non_filtered_reciprocal_mean_rank_tails=non_filtered_mr[4],
            non_filtered_reciprocal_mean_rank_all=non_filtered_mr[5],
            non_filtered_amri_heads=non_filtered_mr[6],
            non_filtered_amri_tails=non_filtered_mr[7],
            non_filtered_amri_all=non_filtered_mr[8],
        )

    @staticmethod
    def write_result_object_to_file(
        file_to_be_written: str,
        result_object: EvaluatorResult,
    ) -> None:
        non_filtered_text = (
            f"\nThis is the evaluation of file {result_object.evaluated_file}\n\n"
            + "Non-filtered Results\n"
            + "--------------------\n"
            + f"Test set size: {result_object.test_set_size}\n"
            + f"Hits at {result_object.n} (Heads): {result_object.non_filtered_hits_at_n_heads}\n"
            + f"Hits at {result_object.n} (Tails): {result_object.non_filtered_hits_at_n_tails}\n"
            + f"Hits at {result_object.n} (All): {result_object.non_filtered_hits_at_n_all}\n"
            + f"Relative Hits at {result_object.n}: {result_object.non_filtered_hits_at_n_relative}\n"
            + f"Mean rank (Heads): {result_object.non_filtered_mean_rank_heads}\n"
            + f"Mean rank (Tails): {result_object.non_filtered_mean_rank_tails}\n"
            + f"Mean rank (All): {result_object.non_filtered_mean_rank_all}\n"
            + f"Mean reciprocal rank (Heads): {result_object.non_filtered_reciprocal_mean_rank_heads}\n"
            + f"Mean reciprocal rank (Tails): {result_object.non_filtered_reciprocal_mean_rank_tails}\n"
            + f"Mean reciprocal rank (All): {result_object.non_filtered_reciprocal_mean_rank_all}\n"
        )

        filtered_text = (
            "\nFiltered Results\n"
            + "----------------\n"
            + f"Test set size: {result_object.test_set_size}\n"
            + f"Hits at {result_object.n} (Heads): {result_object.filtered_hits_at_n_heads}\n"
            + f"Hits at {result_object.n} (Tails): {result_object.filtered_hits_at_n_tails}\n"
            + f"Hits at {result_object.n} (All): {result_object.filtered_hits_at_n_all}\n"
            + f"Relative Hits at {result_object.n}: {result_object.filtered_hits_at_n_relative}\n"
            + f"Mean rank (Heads): {result_object.filtered_mean_rank_heads}\n"
            + f"Mean rank (Tails): {result_object.filtered_mean_rank_tails}\n"
            + f"Mean rank (All): {result_object.filtered_mean_rank_all}\n"
            + f"Mean reciprocal rank (Heads): {result_object.filtered_reciprocal_mean_rank_heads}\n"
            + f"Mean reciprocal rank (Tails): {result_object.filtered_reciprocal_mean_rank_tails}\n"
            + f"Mean reciprocal rank (All): {result_object.filtered_reciprocal_mean_rank_all}\n"
        )

        with open(file_to_be_written, "w+", encoding="utf8") as f:
            f.write(non_filtered_text + "\n")
            f.write(filtered_text)


    @staticmethod
    def write_results_to_file(
        file_to_be_evaluated: str,
        data_set: DataSet,
        file_to_be_written: str = "./results.txt",
    ) -> None:
        """Executes a filtered and non-filtered evaluation and prints the results to the console and to a file.

        Parameters
        ----------
        file_to_be_evaluated : str
            File path to the file that shall be evaluated.
        data_set : DataSet
            The data set that is under evaluation.
        file_to_be_written : str
            File path to the file that shall be written.
        """

        # calculate the results
        results = Evaluator.calculate_results(
            file_to_be_evaluated=file_to_be_evaluated,
            data_set=data_set,
        )

        # write the results to the specified file
        Evaluator.write_result_object_to_file(
            file_to_be_written=file_to_be_written,
            result_object=results,
        )
