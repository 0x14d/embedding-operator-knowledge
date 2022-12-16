import os
from enum import Enum
import io
from typing import List

# making sure that the relative path works
package_directory = os.path.dirname(os.path.abspath(__file__))


class DataSet(Enum):
    """The datasets that are available for evaluation. If a new enum value shall be added, a triple is to be stated with
    [0]: relative test set path
    [1]: relative train set path
    [2]: relative validation set path
    """

    FB15K = (
        os.path.join(
            package_directory, "datasets", "fb15k", "freebase_mtr100_mte100-test.txt"
        ),
        os.path.join(
            package_directory, "datasets", "fb15k", "freebase_mtr100_mte100-train.txt"
        ),
        os.path.join(
            package_directory, "datasets", "fb15k", "freebase_mtr100_mte100-valid.txt"
        ),
    )
    WN18 = (
        os.path.join(package_directory, "datasets", "wn18", "wordnet-mlj12-test.txt"),
        os.path.join(package_directory, "datasets", "wn18", "wordnet-mlj12-train.txt"),
        os.path.join(package_directory, "datasets", "wn18", "wordnet-mlj12-valid.txt"),
    )

    def test_set(self) -> List[List[str]]:
        """Get the parsed test dataset.

        Returns
        -------
        List[List[str]]
            A list of parsed triples.
        """
        return self._parse_tab_separated_data(self.test_set_path())

    def train_set(self) -> List[List[str]]:
        """Get the parsed training dataset.

        Returns
        -------
        List[List[str]]
            A list of parsed triples.
        """
        return self._parse_tab_separated_data(self.train_set_path())

    def valid_set(self) -> List[List[str]]:
        """Get the parsed validation dataset.

        Returns
        -------
        List[List[str]
            A list of parsed triples.
        """
        return self._parse_tab_separated_data(self.valid_set_path())

    @staticmethod
    def _parse_tab_separated_data(file_path) -> List[List[str]]:
        """Parses the given file.

        Parameters
        ----------
        file_path : str
            Path to the file that shall be read. Expected format: token \tab token \tab token

        Returns
        -------
        List[List[str]]
            List of triples.
        """
        result = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.replace("\n", "")
                tokens = line.split(sep="\t")
                result.append(tokens)
        return result

    def test_set_path(self) -> str:
        """Get the test dataset path of the given dataset.

        Returns
        -------
        str
            Relative path to the test dataset.
        """
        return self.value[0]

    def train_set_path(self) -> str:
        """Get the training dataset path of the given dataset.

        Returns
        -------
        str
            Relative path to the training dataset.
        """
        return self.value[1]

    def valid_set_path(self) -> str:
        """Get the validation dataset path of the given dataset.

        Returns
        -------
        str
            Relative path to the validation dataset.
        """
        return self.value[2]

    @staticmethod
    def write_training_file_nt(data_set, file_to_write: str) -> None:
        """File in NT format that can be parsed by jRDF2Vec (https://github.com/dwslab/jRDF2Vec).
        The validation and the training set are combined.

        Parameters
        ----------
        data_set : DataSet
            The dataset that shall be persisted as NT file for vector training.
        file_to_write : str
            The file that shall be written.
        """

        with io.open(file_to_write, "w+", encoding="utf8") as f:
            data_to_write = data_set.train_set()
            data_to_write.extend(data_set.valid_set())
            for triple in data_to_write:
                f.write(
                    "<" + triple[0] + "> <" + triple[1] + "> <" + triple[2] + "> .\n"
                )


if __name__ == "__main__":
    DataSet.write_training_file_nt(DataSet.WN18, "../wordnet_kbc.nt")
    DataSet.write_training_file_nt(DataSet.FB15K, "../fb15k_kbc.nt")
