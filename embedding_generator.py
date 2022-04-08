from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import Tensor

from graph_embeddings.node_embeddings import NodeEmbeddings
from graph_embeddings.subgraph_embedding import SubGraphEmbedding
from schemas import TrainConfig


class EmbeddingGenerator:
    def __init__(self, config: TrainConfig, embedding_version_path=None, influential_only=False, use_head=False):
        if embedding_version_path is None:
            version_path = config.kil_config.embedding_folder
        else:
            version_path = embedding_version_path
        self._node_embeddings = NodeEmbeddings(base_folder=config.main_folder, node_embeddings=version_path,
                                               influential_only=influential_only, use_head=use_head)
        self._vertecies: DataFrame = self._node_embeddings.metadata
        self._edges: DataFrame = self._node_embeddings.edges
        self._skg_lut: DataFrame = pd.read_csv(
            'graph_embeddings/skg_embeddings' + '/skg_embeddings.tsv', sep='\t',
            names=[i for i in range(0, 46)])

    @property
    def node_embeddings(self) -> NodeEmbeddings:
        return self._node_embeddings

    @property
    def edges(self) -> DataFrame:
        return self._edges

    @property
    def vertecies(self) -> DataFrame:
        return self._vertecies

    def get_embedding_from_lut(self, rating) -> Tensor:
        indices: List[int] = self._get_indices_from_rating_names([rating])
        index_list = [idx for idx in self._vertecies.loc[
            (self._vertecies['type'] == 'qual_influence') & (self._vertecies['key'] != 'overall_ok')].index.values]
        return Tensor(self._skg_lut.loc[index_list.index(indices[0])])

    def get_sub_kg_from_ratings(self, ratings: List[str], distance_measure="euclidean", use_head=False) -> Optional[
        SubGraphEmbedding]:
        """
        calculate sub_kg embeddings for every rating in list and concatenate them
        :return: embedding from sub knowledge graph
        """
        if len(ratings) == 0:
            return None
        indices: List[int] = self._get_indices_from_rating_names(ratings)
        edges = DataFrame(np.vstack([self._edges.loc[self._edges['from'] == index] for index in indices]),
                          columns=['from', 'to', 'experiment'])
        if edges.empty:
            return None
        return SubGraphEmbedding(self._node_embeddings, edges, distance_measure, use_head)

    def _get_indices_from_rating_names(self, ratings: List[str]) -> List[int]:
        """
        get indices of the rating names in the vertecies table
        :param ratings:
        :return:
        """
        return [self._vertecies.index[self._vertecies['name'] == rating][0] for rating in ratings]
