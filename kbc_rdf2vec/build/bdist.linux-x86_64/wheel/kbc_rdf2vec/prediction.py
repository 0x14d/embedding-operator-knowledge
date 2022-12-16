import sys
from enum import Enum
from random import randint, random
from typing import List, Tuple, Any
import logging.config
import numpy as np
from gensim.models import KeyedVectors
from tensorflow import keras
from tensorflow.keras import losses
import os

from kbc_rdf2vec.dataset import DataSet

logconf_file = os.path.join(os.path.dirname(__file__), "log.conf")
logging.config.fileConfig(fname=logconf_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class PredictionFunction:
    """Abstract class for prediction functions."""

    def __init__(
        self,
        keyed_vectors: KeyedVectors,
        data_set: DataSet,
        is_reflexive_match_allowed: bool = False,
    ):
        """Constructor

        Parameters
        ----------
        keyed_vectors : KeyedVectors
            The keyed vector instance to be used to make predictions.
        data_set : DataSet
            The data set for which predictions shall be made.
        is_reflexive_match_allowed : bool
            True if it is allowed to predict H in a <H, L, ?> task and T in a <?, L, T> task.
        """
        self._keyed_vectors = keyed_vectors
        self._data_set = data_set
        self._is_reflexive_match_allowed = is_reflexive_match_allowed

        # by default predicates are required, can be overwritten:
        self.requires_predicates = True

    def transform_to_sorted_list(
        self, result_with_confidence, triple: List[str], is_predict_head: bool
    ) -> List[Tuple[str, float]]:
        """If n is none, the result type of the most_similar action in gensim is a numpy array that needs to be mapped
        manually to the vocabulary. This implementation will stop processing the list after the correct solution has
        been found.

        Parameters
        ----------
        result_with_confidence
        triple : List[str]
        is_predict_head: bool
            True if we just predicted the head.

        Returns
        -------
        List[Tuple[str, float]]
        """
        new_result_with_confidence = []
        assert len(result_with_confidence) == len(self._keyed_vectors)
        for i, similarity in enumerate(result_with_confidence):
            word = self._keyed_vectors.index_to_key[i]
            # we do not want to predict the prediacate:
            if word != triple[1]:
                if self._is_reflexive_match_allowed:
                    # we allow for reflexive matches, there are no further restrictions:
                    # (b / c is_reflexive_match_allowed == True).
                    new_result_with_confidence.append((word, similarity))
                elif not is_predict_head and word != triple[0]:
                    # we predict the tail (not head) and the word must not be the head
                    new_result_with_confidence.append((word, similarity))
                elif is_predict_head and word != triple[2]:
                    # we predict the head and the word must not be the tail of the statement
                    # (b/c is_reflexive_match_allowed == True).
                    new_result_with_confidence.append((word, similarity))
            if is_predict_head and word == triple[0]:
                # we found the correct solution, there is no reason to continue here
                break
            elif not is_predict_head and word == triple[2]:
                # we found the correct solution, there is no reason to continue here
                break
        result_with_confidence = sorted(
            new_result_with_confidence, key=lambda x: x[1], reverse=True
        )
        return result_with_confidence

    def predict_heads(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        pass

    def predict_tails(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        pass


class PredictionFunctionEnum(Enum):
    """An enumeration with the implemented similarity functions."""

    MOST_SIMILAR = "most_similar"
    ADDITION = "addition"
    RANDOM = "random"
    PREDICATE_AVERAGING_ADDITION = "predicate_averaging_addition"
    PREDICATE_AVERAGING_MOST_SIMILAR = "predicate_averaging_most_similar"
    ANN = "ann"

    def get_instance(
        self,
        keyed_vectors: KeyedVectors,
        data_set: DataSet,
        is_reflexive_match_allowed: bool = False,
    ) -> PredictionFunction:
        """Obtain the accompanying instance.

        Parameters
        ----------
        data_set: DataSet
            The dataset to be evaluated.
        keyed_vectors
            Keyed vectors instance for which the similarity shall be applied.
        is_reflexive_match_allowed : bool
            True if it is allowed to predict H in a <H, L, ?> task and T in a <?, L, T> task.

        Returns
        -------
        PredictionFunction
            An instance of the PredictionFunctionInterface.
        """
        if self.value == "most_similar":
            return MostSimilarPredictionFunction(
                keyed_vectors=keyed_vectors,
                data_set=data_set,
                is_reflexive_match_allowed=is_reflexive_match_allowed,
            )
        if self.value == "random":
            return RandomPredictionFunction(
                keyed_vectors=keyed_vectors,
                data_set=data_set,
                is_reflexive_match_allowed=is_reflexive_match_allowed,
            )
        if self.value == "predicate_averaging_addition":
            return AveragePredicateAdditionPredictionFunction(
                keyed_vectors=keyed_vectors,
                data_set=data_set,
                is_reflexive_match_allowed=is_reflexive_match_allowed,
            )
        if self.value == "predicate_averaging_most_similar":
            return AveragePredicateMostSimilarPredictionFunction(
                keyed_vectors=keyed_vectors,
                data_set=data_set,
                is_reflexive_match_allowed=is_reflexive_match_allowed,
            )
        if self.value == "addition":
            return AdditionPredictionFunction(
                keyed_vectors=keyed_vectors,
                data_set=data_set,
                is_reflexive_match_allowed=is_reflexive_match_allowed,
            )
        if self.value == "ann":
            return AnnPredictionFunction(
                keyed_vectors=keyed_vectors,
                data_set=data_set,
                is_reflexive_match_allowed=is_reflexive_match_allowed,
            )


class AnnPredictionFunction(PredictionFunction):
    """Artificial Neural Network Approach"""

    def __init__(
        self,
        keyed_vectors: KeyedVectors,
        data_set: DataSet,
        is_reflexive_match_allowed: bool = False,
    ):
        """Constructor. Note that for this prediction function the NNs are immediately learnt.

        Parameters
        ----------
        keyed_vectors : KeyedVectors
            The keyed vector instance to be used to make predictions.
        data_set : DataSet
            The data set for which predictions shall be made.
        is_reflexive_match_allowed : bool
            True if it is allowed to predict H in a <H, L, ?> task and T in a <?, L, T> task.
        """
        super().__init__(
            keyed_vectors=keyed_vectors,
            data_set=data_set,
            is_reflexive_match_allowed=is_reflexive_match_allowed,
        )

        # set this to true to train only with train and to evaluate with the validation set
        self._is_validate = False

        # set ann details here
        batch_size = 1000
        epochs = 15

        # obtain vector dimension (heuristically by determining the dimension of the first vocab entry)
        dimension = len(self._keyed_vectors[self._keyed_vectors.index_to_key[0]])
        input_shape = 2 * dimension

        # set ann details here
        ann_layers = [
            keras.Input(shape=input_shape),
            keras.layers.Dense(dimension),
            keras.layers.Dense(dimension),
        ]

        # required on macOS
        if sys.platform == "darwin":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        # same architecture for both models
        t_prediction_model = keras.Sequential(ann_layers)
        h_prediction_model = keras.Sequential(ann_layers)
        t_prediction_model.compile(loss=losses.mean_squared_error)
        h_prediction_model.compile(loss=losses.mean_squared_error)

        # training
        x_data_t_training = []
        x_data_h_training = []
        y_data_t_training = []
        y_data_h_training = []

        # validation
        x_data_t_validation = []
        y_data_t_validation = []
        x_data_h_validation = []
        y_data_h_validation = []

        # load the training data into the corresponding data structures
        for triple in self._data_set.train_set():
            try:
                h_vector = self._keyed_vectors[triple[0]]
                l_vector = self._keyed_vectors[triple[1]]
                t_vector = self._keyed_vectors[triple[2]]
                x_data_t_training.append(np.append(h_vector, l_vector))
                y_data_t_training.append(np.array(t_vector))
                x_data_h_training.append(np.append(l_vector, t_vector))
                y_data_h_training.append(np.array(h_vector))
            except KeyError:
                logger.error(
                    f"Error linking (training) triple to vectors: \n\t{triple[0]}\t{triple[1]}\t{triple[2]}"
                )
                continue  # ...with the next triple

        if self._is_validate:
            # we do validate, hence we write the data to the validation structure
            for triple in self._data_set.valid_set():
                try:
                    h_vector = self._keyed_vectors[triple[0]]
                    l_vector = self._keyed_vectors[triple[1]]
                    t_vector = self._keyed_vectors[triple[2]]

                    # for predicting t
                    x_data_t_validation.append(np.append(h_vector, l_vector))
                    y_data_t_validation.append(np.array(t_vector))

                    # for predicting h
                    x_data_h_validation.append(np.append(l_vector, t_vector))
                    y_data_h_validation.append(np.array(h_vector))
                except KeyError:
                    logger.error(
                        f"Error linking (validation) triple to vectors: \n\t{triple[0]}\t{triple[1]}\t{triple[2]}"
                    )
                    continue  # ...with the next triple
        else:
            # we do not validate
            # just add the validation data to the training data
            for triple in self._data_set.valid_set():
                try:
                    h_vector = self._keyed_vectors[triple[0]]
                    l_vector = self._keyed_vectors[triple[1]]
                    t_vector = self._keyed_vectors[triple[2]]

                    # for predicting t
                    x_data_t_training.append(np.append(h_vector, l_vector))
                    y_data_t_training.append(np.array(t_vector))

                    # for predicting h
                    x_data_h_training.append(np.append(l_vector, t_vector))
                    y_data_h_training.append(np.array(h_vector))
                except KeyError:
                    logger.error(
                        f"Error linking (validation) triple to vectors: \n\t{triple[0]}\t{triple[1]}\t{triple[2]}"
                    )
                    continue  # ...with the next triple

        x_data_t_training = np.array(x_data_t_training)
        y_data_t_training = np.array(y_data_t_training)
        x_data_h_training = np.array(x_data_h_training)
        y_data_h_training = np.array(y_data_h_training)

        t_prediction_model.fit(
            x=x_data_t_training,
            y=y_data_t_training,
            batch_size=batch_size,
            epochs=epochs,
        )
        h_prediction_model.fit(
            x=x_data_h_training,
            y=y_data_h_training,
            batch_size=batch_size,
            epochs=epochs,
        )

        if self._is_validate:
            # perform the actual validation
            x_data_h_validation = np.array(x_data_h_validation)
            y_data_h_validation = np.array(y_data_h_validation)
            x_data_t_validation = np.array(x_data_t_validation)
            y_data_t_validation = np.array(y_data_t_validation)

            t_score = t_prediction_model.evaluate(
                x_data_t_validation, y_data_t_validation, verbose=0
            )
            h_score = h_prediction_model.evaluate(
                x_data_h_validation, y_data_h_validation, verbose=0
            )

            logger.info(f"T Prediction Score: {t_score}")
            logger.info(f"H Prediction Score: {h_score}")

        self._t_prediction_model = t_prediction_model
        self._h_prediction_model = h_prediction_model

    def predict_heads(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        result = []
        try:
            t_vector = self._keyed_vectors.get_vector(triple[2])
        except KeyError:
            logger.error(f"Could not find the head {triple[2]} in the vector space.")
            return result
        try:
            l_vector = self._keyed_vectors.get_vector(triple[1])
        except KeyError:
            logger.error(
                f"Could not find the predicate {triple[1]} in the averaged vector space."
            )
            return result
        lt_merged_vector = np.array([np.append(l_vector, t_vector)])
        lookup_vector = self._h_prediction_model.predict(lt_merged_vector)
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=[lookup_vector[0]], topn=n
        )
        if n is None:
            return self.transform_to_sorted_list(
                result_with_confidence, triple, is_predict_head=True
            )
        else:
            return result_with_confidence

    def predict_tails(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        """Predict tails.

        Parameters
        ----------
        triple: List[str]
            Triple for which the tails shall be predicted.
        n: Any
            Number of predictions to make. Set to None to obtain all possible predictions.

        Returns
        -------
        List[Tuple[str, float]
            List of predictions with confidences.
        """
        result = []
        try:
            h_vector = self._keyed_vectors.get_vector(triple[0])
        except KeyError:
            logger.error(f"Could not find the head {triple[0]} in the vector space.")
            return result
        try:
            l_vector = self._keyed_vectors.get_vector(triple[1])
        except KeyError:
            logger.error(
                f"Could not find the predicate {triple[1]} in the averaged vector space."
            )
            return result
        lookup_vector = self._t_prediction_model.predict(
            np.array([np.append(h_vector, l_vector)])
        )
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=[lookup_vector[0]], topn=n
        )
        if n is None:
            return self.transform_to_sorted_list(
                result_with_confidence, triple, is_predict_head=False
            )
        else:
            return result_with_confidence


class RandomPredictionFunction(PredictionFunction):
    """This class randomly picks results for h and t."""

    def predict_heads(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        vocab_size = len(self._keyed_vectors)
        if n is None:
            n = vocab_size
        if n > vocab_size:
            logger.error(
                f"n ({n}) > vocab_size ({vocab_size})! Predicting only {vocab_size} concepts."
            )
            n = vocab_size
        result_indices = set()
        if n != vocab_size:
            # run a (rather slow) drawing algorithm
            while len(result_indices) < n:
                result_indices.add(randint(0, vocab_size - 1))
        else:
            # scramble the list
            range_list = range(0, vocab_size - 1)
            result_indices = sorted(range_list, key=lambda x: random())
        result = []
        for index in result_indices:
            result.append((self._keyed_vectors.index_to_key[index], random()))

        sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
        return sorted_result

    def predict_tails(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        """In the random case, there is no difference between predict_heads and predict_tails.

        Parameters
        ----------
        triple: List[str]
            Triple for which the tails shall be predicted.
        n: Any
            Number of predictions to make. Set to None to obtain all possible predictions.

        Returns
        -------
        List[Tuple[str, float]
            List of predictions with confidences. Note that in this case the confidences are random floats.
        """
        return self.predict_heads(triple=triple, n=n)


class MostSimilarPredictionFunction(PredictionFunction):
    """This class simply calls the gensim "most_similar" function with (h,l) to predict t and with (l,t) to predict
    h. It is expected that an embedding for L exists.
    """

    def predict_heads(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=list(triple[1:]), topn=n
        )
        # important: if n is none, the result type of the most_similar action is a numpy array that needs to be
        # mapped manually.
        if n is None:
            return self.transform_to_sorted_list(
                result_with_confidence, triple, is_predict_head=True
            )
        else:
            return result_with_confidence

    def predict_tails(self, triple: List[str], n: int) -> List[Tuple[str, float]]:
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=list(triple[:2]), topn=n
        )
        # important: if n is none, the result type of the most_similar action is a numpy array with confidences that
        # needs to be mapped manually to the vocabulary.
        if n is None:
            return self.transform_to_sorted_list(
                result_with_confidence, triple, is_predict_head=False
            )
        else:
            return result_with_confidence


class AdditionPredictionFunction(PredictionFunction):
    """This class simply calls the gensim "most_similar" function with (H + L) to predict T and with (T - L) to predict
    H. It is expected that an embedding for L exists.
    """

    def predict_heads(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        l_vector = self._keyed_vectors.get_vector(triple[1])
        t_vector = self._keyed_vectors.get_vector(triple[2])
        lookup_vector = t_vector - l_vector

        result_with_confidence = self._keyed_vectors.most_similar(
            positive=[lookup_vector], topn=n
        )
        # important: if n is none, the result type of the most_similar action is a numpy array that needs to be
        # mapped manually.
        if n is None:
            return self.transform_to_sorted_list(
                result_with_confidence, triple, is_predict_head=True
            )
        else:
            return result_with_confidence

    def predict_tails(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        l_vector = self._keyed_vectors.get_vector(triple[1])
        h_vector = self._keyed_vectors.get_vector(triple[0])
        lookup_vector = h_vector + l_vector

        result_with_confidence = self._keyed_vectors.most_similar(
            positive=[lookup_vector], topn=n
        )
        # important: if n is none, the result type of the most_similar action is a numpy array with confidences that
        # needs to be mapped manually to the vocabulary.
        if n is None:
            return self.transform_to_sorted_list(
                result_with_confidence, triple, is_predict_head=False
            )
        else:
            return result_with_confidence


class AveragePredicateAdditionPredictionFunction(PredictionFunction):
    """Prediction function where predicate embeddings are not taken as is but instead given multiple triples <H,L,T>,
    The vector T-H for each triple containing L is averaged to obtain a new vector for L.
    """

    def __init__(
        self,
        keyed_vectors: KeyedVectors,
        data_set: DataSet,
        is_reflexive_match_allowed: bool = False,
    ):
        logger.info("Initializing AveragePredicatePredictionFunction")
        super().__init__(
            keyed_vectors=keyed_vectors,
            data_set=data_set,
            is_reflexive_match_allowed=is_reflexive_match_allowed,
        )

        self.requires_predicates = False

        # now we build a dictionary from predicate to (subject, object)
        all_triples = []
        all_triples.extend(data_set.valid_set())
        all_triples.extend(data_set.train_set())

        p_to_so = {}
        for triple in all_triples:
            if triple[1] not in p_to_so:
                p_to_so[triple[1]] = {(triple[0], triple[2])}
            else:
                p_to_so[triple[1]].add((triple[0], triple[2]))

        self.p_to_mean = {}
        for p, so in p_to_so.items():
            delta_vectors = []
            for s, o in so:
                try:
                    s_vector = self._keyed_vectors.get_vector(s)
                except KeyError:
                    logger.error(
                        f"Could not find S/H concept {s} in the embedding space."
                    )
                    continue
                try:
                    o_vector = self._keyed_vectors.get_vector(o)
                except KeyError:
                    logger.error(
                        f"Could not find O/T concept {o} in the embedding space."
                    )
                    continue
                so_delta = o_vector - s_vector
                delta_vectors.append(so_delta)
            mean_vector = np.mean(delta_vectors, axis=0)
            self.p_to_mean[p] = mean_vector

    def predict_tails(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        """Predictions are made using most_similar(H+L) where L is an averaged vector.

        Parameters
        ----------
        triple : List[str]
            The triple for which the prediction shall be made.
        n : Any
            None if the maximal number of predictions shall be made, else some upper boundary >= 1.

        Returns
        -------
        List[Tuple[str, float]
        """
        result = []
        try:
            h_vector = self._keyed_vectors.get_vector(triple[0])
        except KeyError:
            logger.error(f"Could not find the head {triple[0]} in the vector space.")
            return result
        try:
            l_vector = self.p_to_mean[triple[1]]
        except KeyError:
            logger.error(
                f"Could not find the predicate {triple[1]} in the averaged vector space."
            )
            return result
        lookup_vector = h_vector + l_vector
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=[lookup_vector], topn=n
        )

        # important: if n is none, the result type of the most_similar action is a numpy array with confidences that
        # needs to be mapped manually to the vocabulary.
        if n is None:
            return self.transform_to_sorted_list(
                result_with_confidence, triple, is_predict_head=False
            )
        else:
            return result_with_confidence

    def predict_heads(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        """Predictions are made using most_similar(T-L) where L is an averaged vector.

        Parameters
        ----------
        triple : List[str]
            The triple for which the prediction shall be made.
        n : Any
            None if the maximal number of predictions shall be made, else some upper boundary >= 1.

        Returns
        -------
        List[Tuple[str, float]]
        """
        result = []
        try:
            t_vector = self._keyed_vectors.get_vector(triple[2])
        except KeyError:
            logger.error(f"Could not find the head {triple[2]} in the vector space.")
            return result
        try:
            l_vector = self.p_to_mean[triple[1]]
        except KeyError:
            logger.error(
                f"Could not find the predicate {triple[1]} in the averaged vector space."
            )
            return result
        lookup_vector = t_vector - l_vector
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=[lookup_vector], topn=n
        )

        # important: if n is none, the result type of the most_similar action is a numpy array with confidences that
        # needs to be mapped manually to the vocabulary.
        if n is None:
            return self.transform_to_sorted_list(
                result_with_confidence, triple, is_predict_head=True
            )
        else:
            return result_with_confidence


class AveragePredicateMostSimilarPredictionFunction(PredictionFunction):
    """Prediction function where predicate embeddings are not taken as is but instead given multiple triples <H,L,T>,
    The vector T-H for each triple containing L is averaged to obtain a new vector for L.
    Predictions are made using most_similar(H,L) respectively most_similar(T,L).
    """

    def __init__(
        self,
        keyed_vectors: KeyedVectors,
        data_set: DataSet,
        is_reflexive_match_allowed: bool = False,
    ):
        logger.info("Initializing AveragePredicatePredictionFunction")
        super().__init__(
            keyed_vectors=keyed_vectors,
            data_set=data_set,
            is_reflexive_match_allowed=is_reflexive_match_allowed,
        )

        self.requires_predicates = False

        # now we build a dictionary from predicate to (subject, object)
        all_triples = []
        all_triples.extend(data_set.valid_set())
        all_triples.extend(data_set.train_set())

        p_to_so = {}
        for triple in all_triples:
            if triple[1] not in p_to_so:
                p_to_so[triple[1]] = {(triple[0], triple[2])}
            else:
                p_to_so[triple[1]].add((triple[0], triple[2]))

        self.p_to_mean = {}
        for p, so in p_to_so.items():
            delta_vectors = []
            for s, o in so:
                try:
                    s_vector = self._keyed_vectors.get_vector(s)
                except KeyError:
                    logger.error(
                        f"Could not find S/H concept {s} in the embedding space."
                    )
                    continue
                try:
                    o_vector = self._keyed_vectors.get_vector(o)
                except KeyError:
                    logger.error(
                        f"Could not find O/T concept {o} in the embedding space."
                    )
                    continue
                so_delta = o_vector - s_vector
                delta_vectors.append(so_delta)
            mean_vector = np.mean(delta_vectors, axis=0)
            self.p_to_mean[p] = mean_vector

    def predict_tails(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        """Predictions are made using most_similar(H, L) where L is an averaged vector.

        Parameters
        ----------
        triple : List[str]
            The triple for which the prediction shall be made.
        n : Any
            None if the maximal number of predictions shall be made, else some upper boundary >= 1.

        Returns
        -------
        List[Tuple[str, float]
        """
        result = []
        try:
            h_vector = self._keyed_vectors.get_vector(triple[0])
        except KeyError:
            logger.error(f"Could not find the head {triple[0]} in the vector space.")
            return result
        try:
            l_vector = self.p_to_mean[triple[1]]
        except KeyError:
            logger.error(
                f"Could not find the predicate {triple[1]} in the averaged vector space."
            )
            return result
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=[h_vector, l_vector], topn=n
        )

        # important: if n is none, the result type of the most_similar action is a numpy array that needs to be
        # mapped manually.
        if n is None:
            new_result_with_confidence = []
            assert len(result_with_confidence) == len(self._keyed_vectors)
            for i, similarity in enumerate(result_with_confidence):
                word = self._keyed_vectors.index_to_key[i]
                if word != triple[0]:
                    # avoid predicting the inputs
                    if self._is_reflexive_match_allowed:
                        new_result_with_confidence.append((word, similarity))
                    elif word != triple[1]:
                        new_result_with_confidence.append((word, similarity))
            result_with_confidence = sorted(
                new_result_with_confidence, key=lambda x: x[1], reverse=True
            )
        return result_with_confidence

    def predict_heads(self, triple: List[str], n: Any) -> List[Tuple[str, float]]:
        """Predictions are made using most_similar(T, L) where L is an averaged vector.

        Parameters
        ----------
        triple : List[str]
            The triple for which the prediction shall be made.
        n : Any
            None if the maximal number of predictions shall be made, else some upper boundary >= 1.

        Returns
        -------
        List[Tuple[str, float]]
        """
        result = []
        try:
            t_vector = self._keyed_vectors.get_vector(triple[2])
        except KeyError:
            logger.error(f"Could not find the head {triple[2]} in the vector space.")
            return result
        try:
            l_vector = self.p_to_mean[triple[1]]
        except KeyError:
            logger.error(
                f"Could not find the predicate {triple[1]} in the averaged vector space."
            )
            return result
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=[t_vector, l_vector], topn=n
        )

        # important: if n is none, the result type of the most_similar action is a numpy array that needs to be
        # mapped manually.
        if n is None:
            new_result_with_confidence = []
            assert len(result_with_confidence) == len(self._keyed_vectors)
            for i, similarity in enumerate(result_with_confidence):
                word = self._keyed_vectors.index_to_key[i]
                if word != triple[1]:
                    if self._is_reflexive_match_allowed:
                        new_result_with_confidence.append((word, similarity))
                    elif word != triple[2]:
                        # avoid predicting the inputs
                        new_result_with_confidence.append((word, similarity))
            result_with_confidence = sorted(
                new_result_with_confidence, key=lambda x: x[1], reverse=True
            )
        return result_with_confidence
