import os
from typing import List
import pandas as pd
import pickle
import uuid
import datetime
import traceback
from knowledge_infusion.graph_embeddings.embedding_generator import EmbeddingGenerator
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.config.modules.knowledge_graph_generator_config import KnowledgeGraphGeneratorType
from knowledge_infusion.utils.schemas import TrainConfig
from knowledge_infusion.embedding_evaluation import EmbeddingEvaluation



EMBEDDING_TYPES = ["TransE", "ComplEx", "ComplExLiteral", "RotatE", "DistMult", "DistMultLiteralGated", "BoxE", "rdf2vec"]
HEAD_VALS = [True, False]
DISTANCE_METRICS = ['euclidean, jaccard']
TIMES = 3

class CompareMethods():
    _trainConfig: TrainConfig
    _sgdConfig: SdgConfig
    _embeddingGenerator: EmbeddingGenerator
    _representation: List[str]
    _embeddings: List[str]
    _aggregation: List[str]
    _embedding_type = str


    def __init__(self):
        self._sgdConfig = SdgConfig.create_config('knowledge_infusion/eval_with_synth_data/configs/sdg/default_config_sdg.json')


    def execute_all_methods(self):
        """
        Main method of this class. This generates the synthetic data, embedds it and evaluates the generated embeddings.
        """
        configs = ['knowledge_infusion/eval_with_synth_data/configs/training/default_config_ckl.json']

        if not os.path.isdir("knowledge_infusion/compare_methods/results/"):
            os.makedirs("knowledge_infusion/compare_methods/results/")
        

        # For each available:
        #   - Knowledge Graph Generator
        #   - Embedding Method
        #   - Headstatus
        for i in range(TIMES):
            comp_results = [] # Array to store the results in

            root_folder = "knowledge_infusion/compare_methods/results/iteration" + str(i) + "/"
            if not os.path.isdir(root_folder):
                os.makedirs(root_folder)
            for knowledge_graph_generators in KnowledgeGraphGeneratorType:
                current_kgg_class = knowledge_graph_generators.get_configuration().get_generator_class()
                with open("example_data/kgg_args.pickle", 'rb') as in_f:
                    kgg_args = pickle.load(in_f)
                current_kgg = current_kgg_class(kgg_args)
                for embedding in EMBEDDING_TYPES:
                    for head in HEAD_VALS:
                        try:
                            self._embedding_type = embedding
                            print("________________________________________________")
                            print("Currently: " + knowledge_graph_generators.value)
                            print("Using Embeding Method:" + embedding + " | Head: " + str(head))
                            self._embedding_type = embedding
                            for config in configs:
                                # Skip quantified conditions since matches@k is not a good metric for them
                                if knowledge_graph_generators.value == 'quantified_conditions':
                                    continue
                                kgg_config = KnowledgeGraphGeneratorType(knowledge_graph_generators)
                                self._sgdConfig.knowledge_graph_generator = kgg_config.get_configuration()

                                # Provide train config
                                self._train_config = TrainConfig.parse_file(config)
                                self._train_config.sdg_config = self._sgdConfig
                                self._train_config.id = uuid.uuid1()
                                self._train_config.time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                self._train_config.seed = self._train_config.seed + (i * 5)

                                self._generate_folder_structur(head, root_folder)

                                emb = EmbeddingGenerator(self._train_config, generate_lut=True, embedding_type=embedding, use_head=head, knowledge_graph_generator=current_kgg)
                                
                                matches_at_k_eval(self._train_config, embedding, knowledge_graph_generators, head, comp_results)
                        except Exception as e:
                            print(e)
                            print(traceback.format_exc())
            with open(root_folder + 'results.pickle', 'wb') as out_f:
                pickle.dump(comp_results, out_f)
            print("All embeddings and evaluations succesfully completed. Iteration: " + str(i))



    def _generate_folder_structur(self, head, root_folder):
        """
        Generate folder structure for current project
        """
        
        embedding_id = self._sgdConfig.knowledge_graph_generator.type.value + "_using_" + self._embedding_type + "/"
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

def matches_at_k_eval(config: TrainConfig, embedding, kgg, head, compr):

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

    num_matches_3_euc_whead, num_matches_5_euc_whead = _get_matches('euclidean', head, True, config, embedding)
    num_matches_3_jac_whead, num_matches_5_jac_whead = _get_matches('jaccard', head, True, config, embedding)

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
        temp.set_axis([ matchnames[i]], axis=1, inplace=True)

        splitted = matchnames[i].split('_')
        res = matches_at_k_result(
            embedding,
            kgg,
            splitted[3],
            head,
            temp.mean(),
            temp.std(),
            splitted[2]
        )
        compr.append(res)
        local_results.append(res)
        print("EVAL")
        print(temp.mean())
        print()
    with open(res_file, 'wb') as out_f:
        pickle.dump(local_results, out_f)
    
    print("FINISHED " + embedding + " with " + kgg)

def _get_matches(metric: str, use_head:bool, single_node:bool, config: TrainConfig, emb):
    
    emb_eval = EmbeddingEvaluation(
        distance_measure=metric,
        use_head=use_head,
        ratings=None,
        train_config_obj=config,
        embedding_type=emb
    )

    if single_node:
        emb_eval.generate_rating_param_sub_kg()
    re_3, _, ed_3, _ = emb_eval.calculate_sub_kg_equality(number_nb=3)
    re_5, _, ed_5, _ = emb_eval.calculate_sub_kg_equality(number_nb=5)

    num_matches_3 = emb_eval.get_unord_diff_kg_and_emb(re_3, ed_3)
    num_matches_5 = emb_eval.get_unord_diff_kg_and_emb(re_5, ed_5)
    #q.put([metric, num_matches_3, num_matches_5])
    return num_matches_3, num_matches_5

class matches_at_k_result:
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


if __name__ == "__main__":
    this = CompareMethods()
    this.execute_all_methods()