import os
import sys
import numpy as np
from numpy.typing import ArrayLike
from typing import List

from numpy.typing import NDArray
from pandas import DataFrame
from .node_embeddings import NodeEmbeddings


class SubGraphEmbedding:
    _embedding: ArrayLike

    def __init__(self, n_embeddings: NodeEmbeddings, edges: DataFrame, distance_measure: str, use_head: bool):
        self._node_embeddings: NodeEmbeddings = n_embeddings
        self._sub_kg_node_indices: List[int] = []
        self._edges: DataFrame = edges
        self._distance_measure = distance_measure
        self._use_head: bool = use_head
        self.calculate_sub_kg_embedding()

    @property
    def embedding(self):
        return self._embedding

    @staticmethod
    def jaccard_distance(x, y):
        """
        jaccard distance implementation
        :return:
        """
        if len(x) == len(y):
            enumerator = np.sum([np.min([x[i], y[i]]) for i in range(0, len(x))])
            denominator = np.sum([np.max([x[i], y[i]]) for i in range(0, len(x))])
            distance_sum = np.divide(enumerator, denominator)
            return 1 - distance_sum

    @staticmethod
    def euclidean_distance(x: NDArray, y: NDArray):
        return np.linalg.norm(x - y)

    def calculate_sub_kg_embedding(self):
        """
        sum-based sub knowledge graph embedding by Kursuncu et al.
        calculate embedding from node embedding
        :return:
        """
        embedding_sum = np.zeros((46,))
        for _, row in self._edges.iterrows():
            head_embedding, _ = self._node_embeddings.get_embedding_and_metadata_by_idx(row['from'])
            tail_embedding, _ = self._node_embeddings.get_embedding_and_metadata_by_idx(row['to'])
            if self._distance_measure == "jaccard":
                distance = SubGraphEmbedding.jaccard_distance(head_embedding.to_numpy(), tail_embedding.to_numpy())
            else: # euclidean distance
                distance = SubGraphEmbedding.euclidean_distance(head_embedding.to_numpy(), tail_embedding.to_numpy())
            if self._use_head:
                concatenated_embeddings = np.concatenate((head_embedding.to_numpy(), tail_embedding.to_numpy()), axis=None)
                tensor_product = np.tensordot([concatenated_embeddings], [distance], 0)
                tensor_product = tensor_product.reshape((46,))
            else:
                tensor_product = np.tensordot([tail_embedding.to_numpy()], [distance], 0)
                tensor_product = tensor_product.reshape((46,))
            embedding_sum = embedding_sum + tensor_product
        self._embedding = embedding_sum

    def get_edge_tails_from_sub_kg(self) -> List[float]:
        return list(self._edges['to'])
