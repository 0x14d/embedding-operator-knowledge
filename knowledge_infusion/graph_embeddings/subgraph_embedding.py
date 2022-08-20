from typing import List

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from pandas import DataFrame
from igraph import Graph

from .node_embeddings import NodeEmbeddings


class SubGraphEmbedding:
    _embedding: ArrayLike

    def __init__(self, n_embeddings: NodeEmbeddings, edges: DataFrame, distance_measure: str, use_head: bool, num_of_hops = 1, graph = None, kgtype = 'basic', startnodeid = 0, embedding_dim = 48):
        self._node_embeddings: NodeEmbeddings = n_embeddings
        self._sub_kg_node_indices: List[int] = []
        self._edges: DataFrame = edges
        self._distance_measure = distance_measure
        self._use_head: bool = use_head
        self.start_node_id = startnodeid
        self.graph = graph
        self.dim = embedding_dim

        if kgtype == 'basic' or kgtype == 'unquantified':
            self.calculate_sub_kg_embedding()
        if kgtype == 'quantified_conditions':
            self.calculate_sub_kg_embedding_from_ig(graph, startnodeid, 3, distance_measure, n_embeddings, use_head)
        elif kgtype == 'quantified_parameters_with_literal':
            self.calculate_sub_kg_embedding()
        elif kgtype == 'quantified_parameters_with_shortcut' or kgtype == 'quantified_parameters_without_shortcut':
            self.calculate_sub_kg_embedding_from_ig(graph, startnodeid, 2, distance_measure, n_embeddings, use_head)
        elif kgtype == 'test':
            self.calculate_sub_kg_embedding_from_ig(graph, startnodeid, 1, distance_measure, n_embeddings, use_head)


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
        TODO remove
        :return:
        """
        embedding_sum = np.zeros((self.dim,))
        for _, row in self._edges.iterrows():
            head_embedding, _ = self._node_embeddings.get_embedding_and_metadata_by_idx(row['from'])
            tail_embedding, _ = self._node_embeddings.get_embedding_and_metadata_by_idx(row['to'])
            if self._distance_measure == "jaccard":
                distance = SubGraphEmbedding.jaccard_distance(head_embedding.to_numpy(), tail_embedding.to_numpy())
            else: # euclidean distance
                distance = SubGraphEmbedding.euclidean_distance(head_embedding.to_numpy(), tail_embedding.to_numpy())
            tensor_product = np.tensordot([tail_embedding.to_numpy()], [distance], 0)
            tensor_product = tensor_product.reshape((self.dim,))
            embedding_sum = embedding_sum + tensor_product
        if self._use_head:
            embedding_sum = head_embedding.to_numpy() + embedding_sum
        self._embedding = embedding_sum

    def n_hop_propagation(self, number_of_hops):
        """
        First Step:
            This step does the propagation.
            This uses the hop-array as a state variable and the relevant_vertices-array as the output
            for every hop:
                1. Explore all neighbors of current hop
                2. Add neighbors to next hop
                3. Add newly found vertices in the current hop to the relevant_vertices
                4. current_hop = next hop
        """
        relevant_vertices = set() # Set for all vertices in the subgraph
        hop = [[self.start_node_id]] # Nodes explored in each
        relevant_vertices.add(self.start_node_id)

        # Do the hops
        for current_hop in range(number_of_hops):
            current = []
            for element in hop[current_hop]: # For every vertex at the current hop stage
                # If parameter has been found stop hopping on this branch
                if self._node_embeddings.get_embedding_and_metadata_by_idx(element)[1]['type'] == 'parameter':
                    continue
                nbs =  self.graph.vs[element].neighbors() # Get the neighbors of the vertex
                for nb in nbs:
                    current.append(nb['name']) # Save neighbors
            hop.append(current) # Next hop = found neighbors
            relevant_vertices.update(current)   # Remember all vertices explored
        

        '''
        Second Step:
            Find the relations between the vertices in the subgraph.
            Because this is a directed Graph we can use the directions to make
            sure every edge is uniquely explored, by only using outgoing relations
        '''
        # From all explored vertices get the outgoing edges
        relevant_edges = []
        for element in relevant_vertices:
            relevant_edges += self.graph.vs[element].out_edges()

        return relevant_edges

    
    def calculate_sub_kg_embedding_from_ig(self, graph: Graph, start_node_id: int, number_of_hops: int, distance_measure: str, node_embeddings: NodeEmbeddings, use_head: bool):
        """
        Calculates the Sub Knowledge Graph Embedding for a given knowledge Graph
        
        Parameters:
            graph (igraph.Graph): The Knowledge Graph
            start_node_id (int): Quality for which the Sub Graph should be built
            number_of_hops (int): Number of hops needed for the specific graph type i.e basic->1, quantified_conditions->3
            distance_measure (str): Which distance measurement calculation should be used
            node_embdeddings (NodeEmbeddings): The Node Embeddings
            use_head (bool): Use Head
        """
        embedding_sum = np.zeros((self.dim,))

        relevant_edges = self.n_hop_propagation(number_of_hops)

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
            else: # euclidean distance
                distance = SubGraphEmbedding.euclidean_distance(head_embedding.to_numpy(), tail_embedding.to_numpy())
            tensor_product = np.tensordot([tail_embedding.to_numpy()], [distance], 0)
            tensor_product = tensor_product.reshape((self.dim,))
            embedding_sum = embedding_sum + tensor_product
        if self._use_head:
            embedding_sum = startembedding.to_numpy() + embedding_sum

        self._embedding = embedding_sum

    def get_edge_tails_from_sub_kg(self) -> List[float]:
        """
        :return: list of ids from all tail nodes
        """
        return list(self._edges['to'])
