# pylint: disable=import-error, eval-used

import argparse
from contextlib import redirect_stdout
import datetime
from datetime import timezone, timedelta
import pickle
import os
import traceback
from typing import List
import uuid

import pandas as pd
from tqdm import tqdm

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorType
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_provider import SyntheticDataProvider
import experiment_definition as ed
from graph2rdf.rdf_graph import RDFGraph
from knowledge_infusion.compare_methods.configs.compare_methods_config import \
    CompareMethodsConfig, DataProvider, EvaluationMethod, AmriConfig, MatchesAtKConfig, DebugConfig
from knowledge_infusion.embedding_evaluation import EmbeddingEvaluation
from knowledge_infusion.graph_embeddings.embedding_config import KnowledgeExtractionMethod
from knowledge_infusion.graph_embeddings.embedding_generator import EmbeddingGenerator
from knowledge_infusion.graph_embeddings.embedding_types import EmbeddingType
from knowledge_infusion.graph_embeddings.node_embeddings import NodeEmbeddings
from knowledge_infusion.graph_evaluation.graph_information import generate_graph_information
from knowledge_infusion.rdf2vec.rdf2vec_link_prediction import calculate_amri_hits_at_k_for_rdf2vec
from knowledge_infusion.utils.schemas import TrainConfig
from rule_base import rule_extraction as re


class CompareMethods():
    """
    Class that runs all the required experiments TIMES times.
    """
    _compareMethodsConfig: List[CompareMethodsConfig]
    _trainConfig: TrainConfig
    _sgdConfig: SdgConfig
    _embeddingGenerator: EmbeddingGenerator
    _representation: List[str]
    _embeddings: List[str]
    _aggregation: List[str]
    _embedding_type = str

    def __init__(self, config: CompareMethodsConfig):
        self._compareMethodsConfig = [config]

    def execute_all_methods(self):
        """
        Main method of this class. This generates the synthetic data, embedds it
        and evaluates the generated embeddings.
        """

        # If not existant create results folder
        if not os.path.isdir("knowledge_infusion/compare_methods/results/"):
            os.makedirs("knowledge_infusion/compare_methods/results/")

        if not self._compareMethodsConfig:
            raise ValueError(
                "No run configuration provided to compare_methods.py")

        # For every configuration:
        for run_config in self._compareMethodsConfig:
            self._sgdConfig = SdgConfig.create_config(run_config.sdg_config)

            configs = [run_config.train_config]
            # Create a folder to store this configurations results in
            run_folder = run_config.results_folder + \
                run_config.name + "/"
            if not os.path.isdir(run_folder):
                os.makedirs(run_folder)

            # Create a folder to store the configuration of this experiment in
            config_folder = run_folder + "config/"
            if not os.path.isdir(config_folder):
                os.makedirs(config_folder)

            run_config.save_configuration(
                config_folder + "compare_methods_config.txt")
            with open(config_folder + "sdg_config.json", "w") as out_f:
                out_f.writelines(self._sgdConfig.json())

            # init tqdm limit
            tqdm_total = 0
            for i in range(run_config.times):
                for knowledge_graph_generators in KnowledgeGraphGeneratorType:
                    for embedding in run_config.embedding_types:
                        for head in run_config.head_vals:
                            for config in configs:
                                tqdm_total = tqdm_total + 1

            # For each available:
            #   - Knowledge Graph Generator
            #   - Embedding Method
            #   - Headstatus
            with tqdm(total=tqdm_total) as pbar:
                for i in range(run_config.times):
                    comp_results = []  # Array to store the results in

                    root_folder = run_folder + "iteration" + str(i) + "/"

                    if not os.path.isdir(root_folder):
                        os.makedirs(root_folder)

                    for knowledge_graph_generators in KnowledgeGraphGeneratorType:
                        for embedding in run_config.embedding_types:
                            for head in run_config.head_vals:

                                self._embedding_type = embedding
                                print(
                                    "________________________________________________")
                                print("Currently: " +
                                      knowledge_graph_generators.value)
                                print("Using Embeding Method:" +
                                      embedding + " | Head: " + str(head))

                                for config in configs:
                                    # Skip quantified conditions since matches@k is not a good metric for them
                                    # Also skip quantified conditions for quantified conclusion rules
                                    if run_config.evaluation_method == EvaluationMethod.MatchesAtK or \
                                            run_config.embedding_config.rule_extraction_method.mode == re.KgToRuleMode.FROM_EDGES:
                                        if (
                                            knowledge_graph_generators in [
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_LITERAL,
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT,
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT2,
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT3,
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT,
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT2,
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT3,
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3,
                                                KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3_WITH_LITERAL
                                            ]
                                        ):
                                            print('SKIP QUANTIFIED CONDITIONS!')
                                            pbar.update(1)
                                            continue

                                    # Provide train config
                                    self._train_config = TrainConfig.parse_file(
                                        config)
                                    self._train_config.sdg_config = self._sgdConfig
                                    self._train_config.id = uuid.uuid1()
                                    self._train_config.time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                    self._train_config.seed = self._train_config.seed + \
                                        (i * 5)
                                    self._train_config.embedding_method = embedding
                                    self._train_config.data_provider = 'synthetic' if run_config.data_provider == DataProvider.SYNTHETIC else 'remote'

                                    kgg_config = KnowledgeGraphGeneratorType(
                                        knowledge_graph_generators)
                                    self._sgdConfig.knowledge_graph_generator = kgg_config.get_configuration()

                                    self._generate_folder_structure(
                                        head, root_folder)

                                    redirect_log_file = f'{self._train_config.main_folder}log.txt'
                                    with open(redirect_log_file, 'a', encoding='utf-8', buffering=1) as redirect:
                                        with redirect_stdout(redirect):
                                            try:
                                                if run_config.data_provider == DataProvider.SYNTHETIC:
                                                    data_provider_class = SyntheticDataProvider(
                                                        self._sgdConfig)
                                                else:
                                                    raise TypeError(
                                                        "No DataProvider specified in compare_methods_config")

                                                emb = EmbeddingGenerator(self._train_config, data_provider_class, generate_lut=False,
                                                                         use_head=head, embedding_config=run_config.embedding_config, test_data=True)

                                                if run_config.generate_rdf_graph and embedding == run_config.embedding_types[0] and i == 0 and head:
                                                    print(
                                                        "Generating RDF Graph")
                                                    graph_dir = root_folder + "rdf_graphs/"

                                                    if not os.path.isdir(graph_dir):
                                                        os.makedirs(graph_dir)

                                                    edges = emb._node_embeddings.edges
                                                    meta = emb._node_embeddings.metadata

                                                    if '_with_literal' in knowledge_graph_generators.value:
                                                        rdf_graph = RDFGraph(
                                                            edges, meta, literal_graph=True)
                                                    else:
                                                        rdf_graph = RDFGraph(
                                                            edges, meta, literal_graph=False)

                                                    rdf_graph.save_rdf_graph(
                                                        graph_dir + knowledge_graph_generators.value + ".ttl")

                                                if run_config.evaluation_method == EvaluationMethod.AmriAndHitsAtK:
                                                    node_embeddings: NodeEmbeddings = emb._node_embeddings
                                                    node_embeddings: NodeEmbeddings = emb._node_embeddings
                                                    comp_results.append(
                                                        evaluate_amri_hitsatk(
                                                            node_embeddings)
                                                    )

                                                    if run_config.generate_graph_infos and \
                                                            embedding == run_config.embedding_types[0] \
                                                            and i == 0:
                                                        graph_data_folder = run_folder + "graph_data/"
                                                        if not os.path.isdir(graph_data_folder):
                                                            os.makedirs(
                                                                graph_data_folder)
                                                        generate_graph_information(
                                                            node_embeddings._kgtype,
                                                            node_embeddings.edges,
                                                            node_embeddings.metadata,
                                                            node_embeddings.train_data,
                                                            node_embeddings.test_data,
                                                            graph_data_folder
                                                        )
                                                elif run_config.evaluation_method == EvaluationMethod.MatchesAtK:
                                                    matches_at_k_eval(
                                                        self._train_config, embedding, knowledge_graph_generators, head, comp_results, run_config.embedding_config)
                                                pbar.update(1)
                                            except Exception as exception:
                                                print(exception)
                                                print(traceback.format_exc())
                                                with open(f'{run_config.results_folder}error.txt', 'a', encoding='utf-8') as log:
                                                    time_stamp_end = datetime.datetime.now(
                                                        timezone(timedelta(hours=2)))
                                                    log.write(
                                                        f'{time_stamp_end.strftime("%d-%m %H:%M:%S")} - {self._train_config.main_folder} \n'
                                                    )
                                                pbar.update(1)
                    with open(root_folder + 'results.pickle', 'wb') as out_f:
                        pickle.dump(comp_results, out_f)

    def _generate_folder_structure(self, head, root_folder):
        """
        Generate folder structure for current project
        """

        embedding_id = self._sgdConfig.knowledge_graph_generator.type.value + \
            "_using_" + self._embedding_type + "/"
        main_folder = root_folder + embedding_id
        output_folder = main_folder + "output/"
        dataset_foldeer = main_folder + "dataset/"
        embedding_folder = main_folder + "embedding/"

        if not os.path.isdir(main_folder):
            os.makedirs(main_folder)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        if not os.path.isdir(dataset_foldeer):
            os.makedirs(dataset_foldeer)
        if not os.path.isdir(embedding_folder):
            os.makedirs(embedding_folder)

        self._train_config.main_folder = main_folder
        self._train_config.output_folder = output_folder
        self._train_config.dataset_folder = dataset_foldeer
        self._train_config.embedding_folder = embedding_folder


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def matches_at_k_eval(config: TrainConfig, embedding, kgg, head, compr, embedding_config, embedding_dim=48):

    if head:
        res_file = config.main_folder + "matches_at_k.pickle"
    else:
        res_file = config.main_folder + "matches_at_k_noHead.pickle"
    if os.path.exists(res_file):
        print("Found previous embeddings")
        with open(res_file, 'rb')as f:
            prev = pickle.load(f)
        for element in prev:
            compr.append(element)
        return

    num_matches_3_euc_whead, num_matches_5_euc_whead = _get_matches(
        'euclidean', head, True, config, embedding_config, embedding_dim=embedding_dim)
    num_matches_3_jac_whead, num_matches_5_jac_whead = _get_matches(
        'jaccard', head, True, config, embedding_config, embedding_dim=embedding_dim)

    matches = [
        num_matches_3_jac_whead, num_matches_5_jac_whead,
        num_matches_3_euc_whead, num_matches_5_euc_whead,
    ]
    matchnames = [
        "num_matches_3_jac_whead", "num_matches_5_jac_whead",
        "num_matches_3_euc_whead", "num_matches_5_euc_whead",
    ]
    local_results = []
    for i in range(len(matches)):
        temp = pd.concat([matches[i]], axis=1)
        temp.set_axis([matchnames[i]], axis=1, inplace=True)

        split_by_detail = matchnames[i].split('_')
        res = matches_at_k_result(
            embedding=embedding,
            representation=kgg,
            distance_measure=split_by_detail[3],
            use_head=head,
            mean=temp.mean(),
            std=temp.std(),
            k=split_by_detail[2]
        )
        if "literal" in kgg and not "Literal" in embedding:
            res.mean = [float("NaN")]
            res.std = 0
        compr.append(res)
        local_results.append(res)

        print("EVAL")
        print(temp.mean())
    with open(res_file, 'wb') as out_f:
        pickle.dump(local_results, out_f)

    print("FINISHED " + embedding + " with " + kgg)


def _get_matches(metric: str, use_head: bool, single_node: bool, config: TrainConfig, embedding_config, embedding_dim=48):
    """calculates the matches@k score for an experiment

    Args:
        metric (str): jaccard or euclid currently supporter
        use_head (bool)
        single_node (bool)
        config (TrainConfig): _description_
        emb (str): embedding method used
        embedding_dim (int, optional): Dimensionality of the emebedding. Defaults to 48.

    Returns:
        tuple: score of k=3, and k=5
    """

    emb_eval = EmbeddingEvaluation(
        distance_measure=metric,
        use_head=use_head,
        ratings=None,
        train_config_obj=config,
        embedding_config=embedding_config
    )

    if single_node:
        emb_eval.generate_rating_param_sub_kg()
    re_3, _, ed_3, _ = emb_eval.calculate_sub_kg_equality(number_nb=3)
    re_5, _, ed_5, _ = emb_eval.calculate_sub_kg_equality(number_nb=5)

    num_matches_3 = emb_eval.get_unord_diff_kg_and_emb(re_3, ed_3)
    num_matches_5 = emb_eval.get_unord_diff_kg_and_emb(re_5, ed_5)
    return num_matches_3, num_matches_5


class matches_at_k_result:
    """Dataclass to store the results from matches at k evaluation in, each object generated of this class resembles
    one distungishable experiment. (e.g. embedding-method || representation || head-state || k)
    """
    embedding: str
    representation: str
    distance_measure: str
    use_head: bool
    mean: float
    std: float
    k: int

    def __init__(self, embedding, representation, distance_measure, use_head, mean, std, k):
        self.embedding = embedding
        self.representation = representation
        self.distance_measure = distance_measure
        self.use_head = use_head
        self.mean = mean
        self.std = std
        self.k = k


def evaluate_amri_hitsatk(node_embeddings: NodeEmbeddings):
    """Evaluate the model with the metrics hits@ and AMRI

    Returns:
        dict: a dictionary containing the results
    """

    with open(node_embeddings._base_folder + "model.pickle", 'rb') as fi:
        data = pickle.load(fi)

    if node_embeddings._embedding_type == 'rdf2vec':
        metric_results = calculate_amri_hits_at_k_for_rdf2vec(
            node_embeddings._base_folder)
    else:
        node_embeddings.pykeen_model = data['model']
        node_embeddings.train_data = data['train']
        node_embeddings.test_data = data['test']

        from pykeen.evaluation import RankBasedEvaluator
        from pykeen.evaluation.ranking_metric_lookup import MetricKey
        evaluator = RankBasedEvaluator()

        mapped_triples = node_embeddings.test_data.mapped_triples
        results = evaluator.evaluate(
            model=node_embeddings.pykeen_model,
            mapped_triples=mapped_triples,
            additional_filter_triples=[
                node_embeddings.train_data.mapped_triples]
        )

        # hits at k for 1,5,10 & arithmeticmeanrank
        metrics = ['adjustedarithmeticmeanrankindex',
                   'hits_at_1', 'hits_at_5', 'hits_at_10']
        sides = ['head', 'tail']
        metric_results = {}
        for side in sides:
            for metric in metrics:
                key = MetricKey(metric, side, 'realistic')
                metric_results[(metric, side)] = results.get_metric(key)
    res = {
        'kg_type': node_embeddings._kgtype,
        'method': node_embeddings._embedding_type,
        'results': metric_results
    }
    with open(node_embeddings._base_folder + 'link_prediciton_results.pickle', 'wb') as out_f:
        pickle.dump(res, out_f)

    return res


def arg_checker(args) -> CompareMethodsConfig:
    """Checks the command line arguments for correctness and parses them to a
    config

    Returns:
        CompareMethodsConfig: Config derived from CLAs
    """

    if args.config == "LinkPrediction":
        config = AmriConfig()
    elif args.config == "Matches":
        config = MatchesAtKConfig()
    elif args.config == "Debug":
        config = DebugConfig()

    config.times = args.iterations

    if args.dp is not None:
        config.data_provider = DataProvider(args.dp)

    if args.rule_extraction is not None:
        config.embedding_config.rule_extraction_method = getattr(
            re, args.rule_extraction)

    if args.knowledge_extraction is not None:
        config.embedding_config.knowledge_extraction_method = \
            KnowledgeExtractionMethod(args.knowledge_extraction)

    if args.knowledge_extraction_weight is not None:
        config.embedding_config.knowledge_extraction_weight_function = \
            eval(f'ed.{args.knowledge_extraction_weight}')

    if args.knowledge_extraction_filter is not None:
        config.embedding_config.knowledge_extraction_filter_function = \
            getattr(ed.AdaptiveFilters, args.knowledge_extraction_filter)

    return config


def no_kbc_error(config: CompareMethodsConfig):
    """Explicit check for the kbc libraries and proper warning if not installed

    Args:
        config (CompareMethodsConfig): configuration class
    """
    # The check for the kbc libraries is done explicitly because they are not
    # installable over pip and therefore require manual installation
    if config.evaluation_method == EvaluationMethod.MatchesAtK:
        return True
    if config.evaluation_method == EvaluationMethod.AmriAndHitsAtK and \
            EmbeddingType.Rdf2Vec in config.embedding_types:
        FAIL = '\033[91m'  # console code is displayed red
        ENDC = '\033[0m'  # End red console print
        try:
            import kbc_evaluation
            import kbc_rdf2vec
        except ModuleNotFoundError:
            print(FAIL + "WARNING:")
            print(
                "Your configuration wants to evaluate RDF2Vec-Embeddings with AMRI and Hits@K metrics")
            print("")
            print(
                "In order to do this special libraries not available over pip are required")
            print("Please refer to the get_kbc.md from the documentation.")
            print(ENDC)
            return False
        return True
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help="Configuration to be used for the Programm [LinkPrediction|Matches|Debug]",
                        choices=["LinkPrediction", "Matches", "Debug"],
                        required=True)
    parser.add_argument("--dp",
                        help="Dataprovider used",
                        choices=list(DataProvider),
                        default=None)
    parser.add_argument("--iterations",
                        help="How many iterations should be run",
                        required=True,
                        type=int)
    parser.add_argument("--rule-extraction",
                        help="Rule extraction method used",
                        type=str,
                        choices=['FromEdge', 'BinnedMergedInfluences'],
                        required=False,
                        default=None)
    parser.add_argument("--knowledge-extraction",
                        help="Knowledge extraction method used",
                        type=str,
                        choices=list(KnowledgeExtractionMethod),
                        default=None)
    parser.add_argument("--knowledge-extraction-weight",
                        help="Knowledge extraction weight function used",
                        type=str,
                        default=None)
    parser.add_argument("--knowledge-extraction-filter",
                        help="Knowledge extraction filter function used",
                        type=str,
                        choices=[f.name for f in ed.AdaptiveFilters],
                        default=None)

    args = parser.parse_args()
    config = arg_checker(args)

    if no_kbc_error(config):
        this = CompareMethods(config)
        os.makedirs(config.results_folder, exist_ok=True)
        with open(f'{config.results_folder}log.txt', 'a', encoding='utf-8') as log:
            time_stamp_start = datetime.datetime.now(
                timezone(timedelta(hours=2)))
            log.write(
                f'{time_stamp_start.strftime("%d-%m %H:%M:%S")} - Started {config.name}\n'
            )
        this.execute_all_methods()
        with open(f'{config.results_folder}log.txt', 'a', encoding='utf-8') as log:
            time_stamp_end = datetime.datetime.now(
                timezone(timedelta(hours=2)))
            log.write(
                f'{time_stamp_end.strftime("%d-%m %H:%M:%S")} - Finished {config.name}\n'
            )
