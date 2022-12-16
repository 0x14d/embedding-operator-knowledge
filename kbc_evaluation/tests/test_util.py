import os

from kbc_evaluation.dataset import DataSet

from kbc_evaluation.util import Util


def test_write_sample_predictions():
    test_file_path = "./tests/test_resources/freebase_filtering_example.txt"
    assert os.path.isfile(test_file_path)

    Util.write_sample_predictions(
        prediction_file=test_file_path,
        file_to_be_written="./filtered_test_file.txt",
        data_set=DataSet.FB15K,
        is_apply_filtering=True,
        top_predictions=10,
        number_of_triples=10,
    )
    assert os.path.isfile("./filtered_test_file.txt")
    os.remove("./filtered_test_file.txt")

    Util.write_sample_predictions(
        prediction_file=test_file_path,
        file_to_be_written="./non_filtered_test_file.txt",
        data_set=DataSet.FB15K,
        is_apply_filtering=False,
        top_predictions=10,
        number_of_triples=10,
    )
    assert os.path.isfile("./non_filtered_test_file.txt")
    os.remove("./non_filtered_test_file.txt")
