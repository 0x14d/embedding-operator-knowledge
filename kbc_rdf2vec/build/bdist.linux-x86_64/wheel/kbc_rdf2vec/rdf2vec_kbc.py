import os
import sys

import gensim
import logging.config
from gensim.models import KeyedVectors
from typing import List, Any, Tuple
from tqdm import tqdm

from kbc_rdf2vec.dataset import DataSet
from kbc_rdf2vec.prediction import PredictionFunctionEnum

logconf_file = os.path.join(os.path.dirname(__file__), "log.conf")
logging.config.fileConfig(fname=logconf_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class Rdf2vecKbc:
    def __init__(
        self,
        model_path: str,
        data_set: DataSet,
        n: Any = 10,
        prediction_function: PredictionFunctionEnum = PredictionFunctionEnum.MOST_SIMILAR,
        file_for_predicate_exclusion: str = None,
        is_print_confidences: bool = False,
        is_reflexive_match_allowed: bool = False,
    ):
        """Constructor

        Parameters
        ----------
        model_path : str
            A path to the gensim model file. The file can also be a keyed vector file with ending ".kv".
        data_set : DataSet
            The dataset for which the prediction shall be performed.
        n : Any
            The number of predictions to make for each triple. If you want all predictions, set n to None.
        file_for_predicate_exclusion : str
            The RDF2Vec model learns embeddings for h,l,t but cannot differentiate between them afterwards. Hence,
            when doing predictions for h and t, it may also predict l. If the file used to train the embedding is given
            here, such relations will be removed from the proposal set.
        is_print_confidences : bool
            True if confidences shall be printed into the evaluation file. Default: False.
        is_reflexive_match_allowed : bool
            True if it is allowed to predict H in a <H, L, ?> task and T in a <?, L, T> task.
        """
        if not os.path.isfile(model_path):
            logger.error(
                f"Cannot find file: {model_path}\nCurrent working directory: {os.getcwd()}"
            )

        if model_path.endswith(".kv"):
            logger.info("Gensim vector file detected.")
            self._vectors = KeyedVectors.load(model_path, mmap="r")
        else:
            self._vectors = gensim.models.Word2Vec.load(model_path).wv

        self.n = n
        self.data_set = data_set
        self.test_set = self.data_set.test_set()
        self.is_print_confidences = is_print_confidences
        self._predicates = set()
        if file_for_predicate_exclusion is not None and os.path.isfile(
            file_for_predicate_exclusion
        ):
            self._predicates = self._read_predicates(file_for_predicate_exclusion)
        self._prediction_function = prediction_function.get_instance(
            keyed_vectors=self._vectors,
            data_set=self.data_set,
            is_reflexive_match_allowed=is_reflexive_match_allowed,
        )

    def _read_predicates(self, file_for_predicate_exclusion: str) -> set:
        """Obtain predicates from the given nt file.

        Parameters
        ----------
        file_for_predicate_exclusion : str
            The NT file which shall be checked for predicates.

        Returns
        -------
        set
            A set of predicates (str).
        """
        with open(file_for_predicate_exclusion, "r", encoding="utf8") as f:
            result_set = set()
            for line in f:
                tokens = line.split(sep=" ")
                result_set.add(self.remove_tags(tokens[1]))
            return result_set

    @staticmethod
    def remove_tags(string_to_process: str) -> str:
        """Removes tags around a string. Space-trimming is also applied.

        Parameters
        ----------
        string_to_process : str
            The string for which tags shall be removed.

        Returns
        -------
        str
            Given string without tags.

        """
        string_to_process = string_to_process.strip(" ")
        if string_to_process.startswith("<"):
            string_to_process = string_to_process[1:]
        if string_to_process.endswith(">"):
            string_to_process = string_to_process[: len(string_to_process) - 1]
        return string_to_process

    def predict(self, file_to_write: str) -> None:
        """Performs the actual predictions. A file will be generated.

        Parameters
        ----------
        file_to_write : str
            File that shall be written for further evaluation.
        """
        with open(file_to_write, "w+", encoding="utf8") as f:
            erroneous_triples = 0
            print("Predicting Tails and Heads")

            if self._prediction_function.requires_predicates:
                is_skip_predicate = False
            else:
                is_skip_predicate = True

            with tqdm(total=len(self.test_set), file=sys.stdout) as pbar:
                for triple in self.test_set:
                    if self._check_triple(triple, is_skip_predicate=is_skip_predicate):
                        f.write(f"{triple[0]} {triple[1]} {triple[2]}\n")
                        heads = self._predict_heads(triple)
                        tails = self._predict_tails(triple)
                        f.write(f"\tHeads: {self._prediction_to_string(heads)}\n")
                        f.write(f"\tTails: {self._prediction_to_string(tails)}\n")
                    else:
                        logger.error(f"Could not process the triple: {triple}")
                        erroneous_triples += 1
                    pbar.update(1)

            # logging output for the user
            if erroneous_triples == 0:
                logger.info("Erroneous Triples: " + str(erroneous_triples))
            else:
                logger.error("Erroneous Triples: " + str(erroneous_triples))

    def _prediction_to_string(self, concepts_with_scores) -> str:
        """Transform a prediction to a string.

        Parameters
        ----------
        concepts_with_scores
            The predicted concepts with scores in a list.

        Returns
        -------
        str
            String representation. Depending on the class configuration, confidences are added or not.
        """
        result = ""
        is_first = True
        for c, s in concepts_with_scores:
            if is_first:
                if self.is_print_confidences:
                    result += c + f"_{{{s}}}"
                else:
                    result += c
                is_first = False
            else:
                if self.is_print_confidences:
                    result += f" {c}" + f"_{{{s}}}"
                else:
                    result += f" {c}"
        return result

    def _predict_heads(self, triple: List[str]) -> List:
        """Predicts n heads given a triple.

        Parameters
        ----------
        triple : List[str]
            The triple for which n heads shall be predicted.

        Returns
        -------
        List
            A list of predicted concepts with confidences.
        """
        result_with_confidence = self._prediction_function.predict_heads(triple, self.n)
        result_with_confidence = self._remove_predicates(result_with_confidence)
        return result_with_confidence

    def _predict_tails(self, triple: List[str]) -> List:
        """Predicts n tails given a triple.

        Parameters
        ----------
        triple: List[str]
            The triple for which n tails shall be predicted.

        Returns
        -------
        List
            A list of predicted concepts with confidences.
        """
        result_with_confidence = self._prediction_function.predict_tails(triple, self.n)
        result_with_confidence = self._remove_predicates(result_with_confidence)
        return result_with_confidence

    def _remove_predicates(self, list_to_process: List) -> List[Tuple[str, float]]:
        """From the result list, all predicates are removed and the new list is returned.

        Parameters
        ----------
        list_to_process : List
            List from which the predicates shall be removed.

        Returns
        -------
        List
            New list with removed predicates.
        """
        result = []
        for entry in list_to_process:
            if not entry[0] in self._predicates:
                result.append(entry)
        return result

    def _check_triple(self, triple: List[str], is_skip_predicate: bool = True) -> bool:
        """Triples can only be processed if all three elements are available in the vector space. This methods
        checks for exactly this.

        Parameters
        ----------
        triple : List[str]
            The triple that shall be checked.
        is_skip_predicate : bool
            If True, the predicate will not be checked.

        Returns
        -------
        bool
            True if all three elements of the triple exist in the given vector space, else False.
        """
        try:
            self._vectors.get_vector(triple[0])
            if not is_skip_predicate:
                self._vectors.get_vector(triple[1])
            self._vectors.get_vector(triple[2])
            return True
        except KeyError:
            return False
