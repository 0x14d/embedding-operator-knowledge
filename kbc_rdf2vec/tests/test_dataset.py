from kbc_rdf2vec.dataset import DataSet
import os.path


class TestDataSet:
    def test_files_exist(self):
        """The following is tested
        - Existence of test/valid/train files for every data set in enum DataSet
        """
        for data_set in DataSet:
            test_path = data_set.test_set_path()
            valid_path = data_set.valid_set_path()
            train_path = data_set.train_set_path()
            assert test_path is not None
            assert valid_path is not None
            assert train_path is not None
            assert os.path.isfile(test_path)
            assert os.path.isfile(valid_path)
            assert os.path.isfile(train_path)

    def test_write_training_file_nt(self):
        """Only makes sure that a file is written."""
        for i in DataSet:
            file_to_write = f"./{i}_test_file.nt"
            DataSet.write_training_file_nt(data_set=i, file_to_write=file_to_write)
            assert os.path.isfile(file_to_write)
            with open(file=file_to_write, mode="r", encoding="utf8") as f:
                content = f.readlines()
                content_length = len(content)
                assert content_length > 0  # make sure multiple lines exist
                assert content[10].count("> <") == 2  # random line test
            os.remove(file_to_write)

    def test_file_parser(self):
        """The following is tested
        - whether the given files can be parsed.
        """
        for data_set in DataSet:
            test_data = data_set.test_set()
            self._assert_triples_not_none(test_data)
            train_data = data_set.train_set()
            self._assert_triples_not_none(train_data)
            valid_data = data_set.valid_set()
            self._assert_triples_not_none(valid_data)
            # making sure that different files were read:
            assert (
                test_data[0][0] != train_data[0][0]
                or test_data[0][1] != train_data[0][1]
                or test_data[0][2] != train_data[0][2]
            )
            assert (
                train_data[0][0] != valid_data[0][0]
                or train_data[0][1] != valid_data[0][1]
                or train_data[0][2] != valid_data[0][2]
            )

    @staticmethod
    def _assert_triples_not_none(parsed_triples):
        """Simply runs a couple of assert statements for the given parsed triples.

        Parameters
        ----------
        parsed_triples : List[List[str]]
            The list of triples to be evaluated.

        """
        assert len(parsed_triples) > 10
        assert parsed_triples[0][0] is not None
        assert parsed_triples[0][1] is not None
        assert parsed_triples[0][2] is not None
