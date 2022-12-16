from typing import List

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from igraph import Graph
from sklearn.preprocessing import normalize

from .embedding_config import NormalizationMethods

from .node_embeddings import NodeEmbeddings
from data_provider.knowledge_graphs.config.knowledge_graph_generator_config import KnowledgeGraphGeneratorType
from data_provider.knowledge_graphs.sub_graph_extractor import SubGraphExtractor

class SubGraphEmbedding:
    _embedding: ArrayLike

    def __init__(self, n_embeddings: NodeEmbeddings, distance_measure: str, use_head: bool, graph = None, kgtype = 'basic', startnodeid = 0, embedding_dim = 48, vertices=None, norm_method=NormalizationMethods.number_nodes):
        self._node_embeddings: NodeEmbeddings = n_embeddings
        self._sub_kg_node_indices: List[int] = []
        self._distance_measure = distance_measure
        self._use_head: bool = use_head
        self.start_node_id = startnodeid
        self.graph = graph
        self.dim = embedding_dim

        self.vertices = vertices
        self._norm_method: NormalizationMethods = norm_method
            
        self.kgtype = kgtype
        if isinstance(kgtype, str):
            self.kgtype = KnowledgeGraphGeneratorType(kgtype)
            
        self.sg_extractor = SubGraphExtractor(graph, self.kgtype)

        self.calculate_sub_kg_embedding(graph, startnodeid, distance_measure, n_embeddings, use_head)


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


    def calculate_sub_kg_embedding(self, graph: Graph, start_node_id: int, distance_measure: str, node_embeddings: NodeEmbeddings, use_head: bool):
        """
        Calculates the Sub Knowledge Graph Embedding for a given knowledge Graph
        This method works on graphs created with the Library iGraph.
        
        Parameters:
            graph (igraph.Graph): The Knowledge Graph
            start_node_id (int): Quality for which the Sub Graph should be built
            number_of_hops (int): Number of hops needed for the specific graph type i.e basic->1, quantified_conditions->3
            distance_measure (str): Which distance measurement calculation should be used
            node_embdeddings (NodeEmbeddings): The Node Embeddings
            use_head (bool): Use Head
        """
        embedding_sum = np.zeros((self.dim,))

        relevant_edges, relevant_nodes = self.sg_extractor.get_sub_graph_for_quality(start_node_id)
        number_of_nodes = len(relevant_nodes)
        '''
        Recycled code from above to calculate the change
        '''
        startembedding, _ = node_embeddings.get_embedding_and_metadata_by_idx(start_node_id)
        embedding_sum = np.zeros((self.dim,))
        for row in relevant_edges:
            head_embedding, _ = node_embeddings.get_embedding_and_metadata_by_idx(row.source)
            tail_embedding, _ = node_embeddings.get_embedding_and_metadata_by_idx(row.target)
            if distance_measure == "jaccard":
                distance = SubGraphEmbedding.jaccard_distance(head_embedding.to_numpy(), tail_embedding.to_numpy())
            elif distance_measure == "euclidean":
                distance = SubGraphEmbedding.euclidean_distance(head_embedding.to_numpy(), tail_embedding.to_numpy())
            else:
                raise NotImplementedError()
            tensor_product = np.tensordot([tail_embedding.to_numpy()], [distance], 0)
            tensor_product = tensor_product.reshape((self.dim,))
            embedding_sum = embedding_sum + tensor_product
        if self._use_head:
            embedding_sum = startembedding.to_numpy() + embedding_sum
        else:
            number_of_nodes = number_of_nodes - 1

        self._embedding = embedding_sum
        if self._norm_method == NormalizationMethods.number_nodes:
            self._embedding = embedding_sum/number_of_nodes
        elif self._norm_method == NormalizationMethods.unit_norm:
            if self._embedding.dtype != 'complex128':
                self._embedding = normalize(embedding_sum/number_of_nodes)
            else:
                raise ValueError("Complex values cannot be normalized")
        elif self._norm_method == NormalizationMethods.none:
            self._embedding = embedding_sum
        else:
            raise ValueError("Unknown NormalizationMethod!")

    def get_parameters_in_sub_kg(self) -> List[int]:
        """Gets the IDs of all Parameters found in this SubKG

        Returns:
            List[int]: IDs of all Parameters
        """
        return self.sg_extractor.get_tails()
