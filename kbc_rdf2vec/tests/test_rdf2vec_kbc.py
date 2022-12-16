import os

from kbc_rdf2vec.dataset import DataSet
from kbc_rdf2vec.rdf2vec_kbc import Rdf2vecKbc
from kbc_rdf2vec.prediction import PredictionFunctionEnum


class TestRdf2vecKbc:
    def test_remove_tags(self):
        print(Rdf2vecKbc.remove_tags("<hello>"))
        assert Rdf2vecKbc.remove_tags("<hello>") == "hello"
        assert Rdf2vecKbc.remove_tags(" <hello>  ") == "hello"

    def test_predict(self):
        # with n = 10
        test_file_path = "./tests/test_resources/wn_test_model.kv"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        kbc = Rdf2vecKbc(
            data_set=DataSet.WN18,
            model_path=test_file_path,
            n=10,
            is_print_confidences=False,
        )
        file_to_write = "./test_prediction.txt"
        kbc.predict(file_to_write=file_to_write)
        assert os.path.isfile(file_to_write)
        with open(file_to_write, "r", encoding="utf8") as f:
            n_10_content = f.read()
            assert n_10_content.count("_member_meronym") > 0
            assert n_10_content.count("_{0") == 0
        os.remove(file_to_write)

        # with n = None
        kbc = Rdf2vecKbc(
            data_set=DataSet.WN18,
            model_path=test_file_path,
            n=None,
            is_print_confidences=False,
        )
        kbc.predict(file_to_write=file_to_write)
        assert os.path.isfile(file_to_write)
        with open(file_to_write, "r", encoding="utf8") as f:
            n_none_content = f.read()
            assert n_none_content.count("_member_meronym") > 0
            assert n_10_content.count("_{0") == 0

    def test_with_confidence_printing(self):
        # with n = 10
        test_file_path = "./tests/test_resources/wn_test_model.kv"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        kbc = Rdf2vecKbc(
            data_set=DataSet.WN18,
            model_path=test_file_path,
            n=10,
            is_print_confidences=True,
        )
        file_to_write = "./test_prediction.txt"
        kbc.predict(file_to_write=file_to_write)
        assert os.path.isfile(file_to_write)
        with open(file_to_write, "r", encoding="utf8") as f:
            n_10_content = f.read()
            assert n_10_content.count("_member_meronym") > 0
            assert n_10_content.count("_{0") > 10
        os.remove(file_to_write)

        # with n = None
        kbc = Rdf2vecKbc(
            data_set=DataSet.WN18,
            model_path=test_file_path,
            n=None,
            is_print_confidences=True,
        )
        kbc.predict(file_to_write=file_to_write)
        assert os.path.isfile(file_to_write)
        with open(file_to_write, "r", encoding="utf8") as f:
            n_none_content = f.read()
            assert n_none_content.count("_member_meronym") > 0
            assert n_10_content.count("_{0") > 10

    def test_predict_with_relation_filter(self):
        """Tests whether relations are excluded from proposals."""
        model_file_path = "./tests/test_resources/wn_test_model.kv"
        nt_file_path = "./tests/test_resources/wn_test.nt"
        if not os.path.isfile(model_file_path):
            os.chdir("./..")
        kbc = Rdf2vecKbc(
            data_set=DataSet.WN18,
            model_path=model_file_path,
            file_for_predicate_exclusion=nt_file_path,
        )
        file_to_write = "./test_prediction.txt"
        kbc.predict(file_to_write=file_to_write)
        assert os.path.isfile(file_to_write)
        with open(file_to_write, "r", encoding="utf8") as f:
            content = f.read()
            assert content.count("_member_meronym") == 0
        os.remove(file_to_write)

    def test_prediction_function_ann(self):
        """Tests whether relations are excluded from proposals and whether ANN works."""
        model_file_path = "./tests/test_resources/wn_test_model.kv"
        nt_file_path = "./tests/test_resources/wn_test.nt"
        if not os.path.isfile(model_file_path):
            os.chdir("./..")
        kbc = Rdf2vecKbc(
            data_set=DataSet.WN18,
            model_path=model_file_path,
            file_for_predicate_exclusion=nt_file_path,
            prediction_function=PredictionFunctionEnum.ANN,
        )
        file_to_write = "./test_ann_prediction.txt"
        kbc.predict(file_to_write=file_to_write)
        assert os.path.isfile(file_to_write)
        with open(file_to_write, "r", encoding="utf8") as f:
            content = f.read()
            assert content.count("_member_meronym") == 0
        os.remove(file_to_write)
