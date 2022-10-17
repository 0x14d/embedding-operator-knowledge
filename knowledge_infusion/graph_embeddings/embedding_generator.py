from typing import List, Optional

import numpy as np
import pandas as pd
import os
from pandas import DataFrame
from torch import Tensor, zeros
from igraph import Graph

from .node_embeddings import NodeEmbeddings
from .subgraph_embedding import SubGraphEmbedding
from ..utils.schemas import TrainConfig


class EmbeddingGenerator:
    # TODO data_provider
    def __init__(self, config: TrainConfig, embedding_version_path=None, influential_only=False, use_head=False, generate_lut=True, embedding_type="TransH", knowledge_graph_generator= None, embedding_dim=48, rdf2vec_config = None):
        self.kgtype = config.sdg_config.knowledge_graph_generator.type.value
        self.use_head = use_head
        self.embedding_dim = embedding_dim
        if embedding_version_path is None:
            version_path = config.kil_config.embedding_folder
        else:
            version_path = embedding_version_path
        self._node_embeddings = NodeEmbeddings(
            base_folder=config.embedding_folder,
            node_embeddings=version_path,
            influential_only=influential_only,
            use_head=use_head,
            type=embedding_type,
            kg_type=self.kgtype,
            random_seed=config.seed,
            knowledge_graph_generator=knowledge_graph_generator,
            embedding_dim=embedding_dim,
            rdf2vec_config=rdf2vec_config
        )
        self._vertecies: DataFrame = self._node_embeddings._metadata
        self._edges: DataFrame = self._node_embeddings._edges
        self.ig = Graph.DataFrame(self.node_embeddings.edges, directed = True)
        #self._skg_lut: DataFrame = pd.read_csv(
        #    'knowledge_infusion/graph_embeddings/skg_embeddings' + '/skg_embeddings.tsv', sep='\t',
        #    names=[i for i in range(0, 46)])
        if generate_lut:
            self._skg_lut: DataFrame = self._load_lut(config.embedding_folder)

    @property
    def node_embeddings(self) -> NodeEmbeddings:
        return self._node_embeddings

    @property
    def edges(self) -> DataFrame:
        return self._edges

    @property
    def vertecies(self) -> DataFrame:
        return self._vertecies

    def _generate_lut(self, file, file2):
        if self.kgtype == 'basic' or self.kgtype == 'unquantified':
            skg_lut = DataFrame(np.vstack([SubGraphEmbedding(self.node_embeddings, self.edges.loc[self.edges['from'] == idx], "jaccard", False, embedding_dim=self.embedding_dim).embedding.T for idx in self.vertecies.loc[self.vertecies['type'] == 'qual_influence'].index.values]))
        elif self.kgtype == 'quantified_parameters_with_literal':
            relevant_edges = self.edges.loc[self.edges['literal_included'] == 'None']
            skg_lut = DataFrame(np.vstack([SubGraphEmbedding(self.node_embeddings, relevant_edges[relevant_edges['from'] == idx], "jaccard", False, embedding_dim=self.embedding_dim).embedding.T for idx in self.vertecies.loc[self.vertecies['type'] == 'qual_influence'].index.values]))
        elif 'quantified_param' in self.kgtype:
            skg_lut = DataFrame(np.vstack([SubGraphEmbedding(self.node_embeddings, None, 'jaccard', False, 2, self.ig, self.kgtype, idx, embedding_dim=self.embedding_dim).embedding.T for idx in self.vertecies.loc[self.vertecies['type'] == 'qual_influence'].index.values]))
        elif self.kgtype == 'quantified_conditions':
            skg_lut = DataFrame(np.vstack([SubGraphEmbedding(self.node_embeddings, None, 'jaccard', False, 3, self.ig, 'quantified_conditions', idx, embedding_dim=self.embedding_dim).embedding.T for idx in self.vertecies.loc[self.vertecies['type'] == 'qual_influence'].index.values]))
 
        with open(file, 'w') as out_file:
            csv_str = skg_lut.to_csv(index=False, sep='\t', index_label=False, header=False)
            out_file.write(csv_str)

        skg_meta = DataFrame(np.vstack(self.vertecies.loc[self.vertecies['type'] == 'qual_influence'].name))
        with open(file2, 'w') as out_file:
            csv_str = skg_meta.to_csv(index=False, sep='\t', index_label=False, header=False)
            out_file.write(csv_str)
        return skg_lut

    def _load_lut(self, folder):
        if self.use_head:
            file = folder + 'graph_embeddings/skg_embeddings.tsv'
            file2 = folder + 'graph_embeddings/skg_metadata.tsv'
        else:
            file = folder + 'graph_embeddings/skg_embeddings_noHead.tsv'
            file2 = folder + 'graph_embeddings/skg_metadata_noHead.tsv'
        if os.path.isfile(file):
            return pd.read_csv(file, sep='\t', names=[i for i in range(0, self.embedding_dim)])
        else:
            return self._generate_lut(file, file2)
    
    def get_embedding_from_lut(self, rating, emb_dim) -> Tensor:
        try:
            indices: List[int] = self._get_indices_from_rating_names([rating])
            index_list = [idx for idx in self._vertecies.loc[
                (self._vertecies['type'] == 'qual_influence') & (self._vertecies['name'] != 'overall_ok')].index.values]
            emb = Tensor(self._skg_lut.loc[index_list.index(indices[0])])
        except IndexError:
            emb = zeros(emb_dim)
        return emb

    def get_sub_kg_from_ratings(self, ratings: List[str], distance_measure="euclidean", use_head=False) -> Optional[
        SubGraphEmbedding]:
        """
        calculate sub_kg embeddings for every rating in list and concatenate them
        :return: embedding from sub knowledge graph
        """
        indices: List[int] = []
        for rating in ratings:
            try:
                indices.extend(self._get_indices_from_rating_names([rating]))
            except:
                pass

        if len(indices) == 0:
            return None
        if self.kgtype == 'basic' or self.kgtype == 'unquantified':
            edges = DataFrame(np.vstack([self._edges.loc[self._edges['from'] == index] for index in indices]),
                            columns=['from', 'to', 'rel', 'experiment'])
            if edges.empty:
                return None
            return SubGraphEmbedding(self._node_embeddings, edges, distance_measure, use_head, embedding_dim=self.embedding_dim)
        
        elif self.kgtype == 'quantified_parameters_with_literal':
            edges = DataFrame(np.vstack([self._edges.loc[self._edges['from'] == index] for index in indices]),
                            columns=['from', 'to', 'rel', 'literal_included'])
            relevant_edges = edges.loc[edges['literal_included'] == 'None']
            return SubGraphEmbedding(self._node_embeddings, relevant_edges, distance_measure, use_head, embedding_dim=self.embedding_dim)

        elif self.kgtype == 'quantified_conditions':
            edges = DataFrame(np.vstack([self._edges.loc[self._edges['to'] == index] for index in indices]),
                            columns=['from', 'to', 'rel'])
            if edges.empty:
                return None
            return SubGraphEmbedding(self._node_embeddings, edges, distance_measure, False, 3, self.ig, 'quantified_conditions', indices[0], embedding_dim=self.embedding_dim)
        
        elif self.kgtype == 'quantified_parameters_without_shortcut' or self.kgtype == 'quantified_parameters_with_shortcut':
            # Todo replace with proper Edge generation from within SubGraphEmbedding
            # 
            edges_to_number = DataFrame(np.vstack([self._edges.loc[self._edges['from'] == index] for index in indices]),
                            columns=['from', 'to', 'rel', 'literal_included'])
            if edges_to_number.empty:
                return None
            number_ids = []
            for edge in edges_to_number.iterrows():
                number_ids.append(edge[1].loc['to'])
            edges = DataFrame(np.vstack([self._edges.loc[self._edges['from'] == index] for index in number_ids]),
                            columns=['from', 'to', 'rel', 'literal_included'])
            if edges.empty:
                return None
            return SubGraphEmbedding(self.node_embeddings, edges, distance_measure, use_head, 2, self.ig, self.kgtype, indices[0], embedding_dim=self.embedding_dim)


    def _get_indices_from_rating_names(self, ratings: List[str]) -> List[int]:
        """
        get indices of the rating names in the vertecies table
        :param ratings:
        :return:
        """
        return [self._vertecies.index[self._vertecies['name'] == rating][0] for rating in ratings]
