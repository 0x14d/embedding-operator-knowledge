"""
Evaluate knowledge graph embedding method
"""
import ast
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import spatial
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.ratings_to_param_dataset import RatingsToParamDataset
from dataset_handler import DatasetHandler
from embedding_generator import EmbeddingGenerator
from graph_embeddings.subgraph_embedding import SubGraphEmbedding
from schemas import TrainConfig


class EmbeddingEvaluation:

    def __init__(self, distance_measure: str, use_head: bool, path='config/default_config.json',
                 influential_only=False):
        """
        Evaluation method with the following steps:
            1. For Every rating in ratings:
                1. Calculate connected parameters in KG (done)
                2. Rank other ratings with similarity from parameter intersections
                3. Rank subgraph embeddings with distance to other ratings
        """
        self._dis_meas = distance_measure
        self._use_head = use_head
        config_path: str = path
        config: TrainConfig = TrainConfig.parse_file(config_path)
        self._emb_generator = EmbeddingGenerator(config, influential_only=influential_only, use_head=use_head)
        self.verts = self._emb_generator.vertecies
        self._ratings = [idx for idx in self.verts[self.verts['type'] == 'qual_influence']['key'] if
                         idx != 'overall_ok']  # all ratings without the overall_ok rating
        params = [idx for idx in self.verts[self.verts['type'] == 'parameter']['key'].index]
        self._ratings_params: Optional[DataFrame] = None

    def generate_rating_param_sub_kg(self):
        """
        Calculate connected parameters in KG for every rating
        :return:
        """
        verts = self._emb_generator.vertecies
        rating_param_dict = {}
        all_params = [idx for idx in self.verts[self.verts['type'] == 'parameter']['key'].index]
        for rating in self._ratings:
            subkg_emb = self._emb_generator.get_sub_kg_from_ratings([rating], distance_measure=self._dis_meas,
                                                                    use_head=self._use_head)
            param_dict = {}
            if subkg_emb is not None:
                params = subkg_emb.get_edge_tails_from_sub_kg()
                for p in all_params:
                    if p in params:
                        param_dict[p] = True
                    else:
                        param_dict[p] = False
            rating_param_dict[rating] = param_dict
        self._ratings_params = DataFrame.from_dict(rating_param_dict, orient="index", columns=all_params)

    def get_number_of_equal_params(self, rating_1: str, rating_2: str) -> int:
        return len(list(filter(lambda val: val is True,
                               self._ratings_params.loc[rating_1].eq(
                                   self._ratings_params.loc[rating_2]))))

    def get_distance_between_embeddings(self, ratings_1: List[str], ratings_2: List[str], eval_meas='euclidean',
                                        distance="jaccard", use_head=False) -> float:
        embedding_1 = self._emb_generator.get_sub_kg_from_ratings(ratings_1, distance_measure=distance,
                                                                  use_head=use_head).embedding
        embedding_2 = self._emb_generator.get_sub_kg_from_ratings(ratings_2, distance_measure=distance,
                                                                  use_head=use_head).embedding
        if eval_meas == 'euclidean':
            return spatial.distance.euclidean(embedding_1, embedding_2)
        else:
            return spatial.distance.cosine(embedding_1, embedding_2)

    def calculate_sub_kg_equality(self, eval_meas='euclidean', number_nb=None) -> (
            DataFrame, DataFrame, DataFrame, DataFrame):
        """

        :param eval_meas:
        :param number_nb: number of neighbours
        :return:
        """
        subkg_equality: dict = {}
        subkg_equality_values: dict = {}
        embedding_distance: dict = {}
        embedding_distance_val: dict = {}
        ratings = list(self._ratings_params.index.values)
        for rating in tqdm(ratings):
            rating_equality: dict = {}
            rating_distance: dict = {}
            for rating2 in ratings:
                # ignore own number of equal params
                if rating != rating2:
                    number_of_equal_params = self.get_number_of_equal_params(rating, rating2)
                    rating_equality[rating2] = number_of_equal_params
                    distance = self.get_distance_between_embeddings(rating.split('-'), rating2.split('-'),
                                                                    eval_meas=eval_meas, distance=self._dis_meas,
                                                                    use_head=self._use_head)
                    rating_distance[rating2] = distance
            # sorted_rating_equality = sorted(rating_equality, key=rating_equality.get, reverse=True)
            sorted_rating_equality = sorted(rating_equality.items(), key=lambda item: item[1], reverse=True)
            sorted_rating_equality_names = [sorted_rating_equality[i][0] for i in range(len(sorted_rating_equality))]
            sorted_rating_equality_values = [round(sorted_rating_equality[i][1],2) for i in range(len(sorted_rating_equality))]
            # sorted_rating_distance = sorted(rating_distance, key=rating_distance.get, reverse=False)
            sorted_rating_distance = sorted(rating_distance.items(), key=lambda item: item[1], reverse=False)
            sorted_rating_distance_names = [sorted_rating_distance[i][0] for i in range(len(sorted_rating_distance))]
            sorted_rating_distance_values = [round(sorted_rating_distance[i][1],2) for i in range(len(sorted_rating_distance))]
            if number_nb is None:
                subkg_equality[rating] = sorted_rating_equality_names
                subkg_equality_values[rating] = sorted_rating_equality_values
                embedding_distance[rating] = sorted_rating_distance_names
                embedding_distance_val[rating] = sorted_rating_distance_values
            else:
                subkg_equality[rating] = sorted_rating_equality_names[:number_nb]
                subkg_equality_values[rating] = sorted_rating_equality_values[:number_nb]
                embedding_distance[rating] = sorted_rating_distance_names[:number_nb]
                embedding_distance_val[rating] = sorted_rating_distance_values[:number_nb]
        subkg_equality_df: DataFrame = DataFrame.from_dict(subkg_equality)
        subkg_equality_values_df: DataFrame = DataFrame.from_dict(subkg_equality_values)
        embedding_distance_df: DataFrame = DataFrame.from_dict(embedding_distance)
        embedding_distance_values_df: DataFrame = DataFrame.from_dict(embedding_distance_val)
        return subkg_equality_df.T, subkg_equality_values_df.T, embedding_distance_df.T, embedding_distance_values_df.T

    def get_ord_diff_kg_and_emb(self, kg_df: DataFrame, emb_df: DataFrame) -> DataFrame:
        diff_df = kg_df.eq(emb_df)
        difference_kg_and_embedding_apply = diff_df.T.apply(pd.value_counts)
        difference_kg_and_embedding_apply.rename(index={True: "#Parameter Match"}, inplace=True)
        difference_kg_and_embedding_apply.fillna(0, inplace=True)
        return DataFrame(difference_kg_and_embedding_apply.loc["#Parameter Match"])
        # return difference_kg_and_embedding.apply(pd.value_counts)

    def get_number_equal_ratings(self, kg_ratings: List[str], emb_ratings: List[str]) -> int:
        return len([value for value in kg_ratings if value in emb_ratings])

    def get_number_of_occurence(self, ratings: List[str], rating_values: DataFrame) -> DataFrame:
        rating_histo = {}
        for rating in ratings:
            rating_idx = rating_values.index[rating_values['key'] == rating][0]
            rating_histo[rating] = len(ast.literal_eval(rating_values.at[rating_idx, 'value']).items())
        return DataFrame.from_dict(rating_histo, orient='index', columns=['occurence'])

    def get_unord_diff_kg_and_emb(self, kg_df: DataFrame, emb_df: DataFrame) -> DataFrame:
        diff_kg_emb_unord = DataFrame(index=kg_df.index)
        for i, row in kg_df.iterrows():
            diff_kg_emb_unord.at[i, 0] = self.get_number_equal_ratings(list(kg_df.loc[i].values),
                                                                       list(emb_df.loc[i].values))
        return diff_kg_emb_unord

    def get_graph_data(self):
        dp = DataProvider(url=DataProvider.remote_url, username=DataProvider.remote_usr,
                          password=DataProvider.remote_pw)
        data, lok, lov, boolean_parameters, returned_graphs, experiment_series_main, label_encoder = \
            dp.get_experiments_with_graphs(influential_influences_only=False)

        weight_methods = [ed.WeightingClustering.CLUSTERING_PARAMETERS_QUALITY]
        filter_methods = [ed.AdaptiveFilters.MEAN]
        aggregated_graphs = GraphAggregation.get_aggregated_graphs(returned_graphs, weight_methods, filter_methods,
                                                                   experiment_series_main, dp.get_edges_dict(),
                                                                   label_encoder)
        graph = aggregated_graphs['clustering_parameters_quality#mean']['graph']

        return kg.prepare_dfs_of_experiments(graph, experiment_series_main, label_encoder, fill_nones=False)

    def get_idx_from_rating(self, rating: str) -> int:
        metadata = pd.read_csv("output.csv")
        return metadata.loc[metadata['key'] == rating].index.values[0]

    def get_rating_from_idx(self, idx: int) -> str:
        metadata = pd.read_csv("output.csv")
        return metadata.loc[idx]['key']

    def generate_rating_param_sub_kg_prop(self):
        dataset_handler = DatasetHandler('ratings2param')
        exp_dataset = RatingsToParamDataset(dataset_handler)
        data_loader: DataLoader = DataLoader(exp_dataset, batch_size=1, shuffle=False)
        all_params = [idx for idx in self.verts[self.verts['type'] == 'parameter']['key'].index]
        rating_param_dict = {}
        for data in data_loader:
            inputs, _ = data
            # ke_batch: Tensor = torch.empty(self._batch_size, 1, 1, self._output_dim)
            for _, feature in enumerate(inputs):
                # choose rating with the highest occurence score
                feature = feature.detach().numpy().flatten()
                rating_list = np.array(self._ratings)
                ratings: List[str] = list(rating_list[np.where(feature > 0)])
                ratings_name = '-'.join(ratings)
                # if ratings_name not in self._ratings_params.index:
                subgraph_emb: Optional[SubGraphEmbedding] = self._emb_generator.get_sub_kg_from_ratings(ratings,
                                                                                                        distance_measure=self._dis_meas,
                                                                                                        use_head=self._use_head)
                if subgraph_emb is not None:
                    params = subgraph_emb.get_edge_tails_from_sub_kg()
                    param_dict = {}
                    for p in all_params:
                        if p in params:
                            param_dict[p] = True
                        else:
                            param_dict[p] = False
                    rating_param_dict[ratings_name] = param_dict
        self._ratings_params = DataFrame.from_dict(rating_param_dict, orient="index", columns=all_params)

    def get_number_of_occurence_prop(self):
        dataset_handler = DatasetHandler('ratings2param')
        exp_dataset = RatingsToParamDataset(dataset_handler)
        data_loader: DataLoader = DataLoader(exp_dataset, batch_size=1, shuffle=False)
        rating_histo = {}
        for data in data_loader:
            inputs, _ = data
            for _, feature in enumerate(inputs):
                feature = feature.detach().numpy().flatten()
                rating_list = np.array(self._ratings)
                ratings: List[str] = list(rating_list[np.where(feature > 0)])
                ratings_name = '-'.join(ratings)
                if ratings_name in rating_histo:
                    rating_histo[ratings_name] += 1
                else:
                    rating_histo[ratings_name] = 1
        return DataFrame.from_dict(rating_histo, orient='index', columns=['occurence'])
