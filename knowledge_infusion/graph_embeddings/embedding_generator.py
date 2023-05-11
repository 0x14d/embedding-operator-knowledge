# pylint: disable=import-error

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import Tensor, zeros  # pylint: disable=no-name-in-module

from data_provider.abstract_data_provider import AbstractDataProvider
from knowledge_infusion.graph_embeddings.embedding_types import EmbeddingType
from knowledge_infusion.graph_embeddings.embedding_config import EmbeddingConfig
from knowledge_infusion.graph_embeddings.node_embeddings import NodeEmbeddings
from knowledge_infusion.graph_embeddings.subgraph_embedding import SubGraphEmbedding
from knowledge_infusion.graph_embeddings.utils.knowledge_graph_import import import_knowledge_graph
from knowledge_infusion.utils.schemas import TrainConfig


class EmbeddingGenerator:
    def __init__(
        self,
        config: TrainConfig,
        data_provider: AbstractDataProvider,
        influential_only=False,
        use_head=False,
        generate_lut=True,
        embedding_config: Optional[EmbeddingConfig] = None,
        test_data=False
    ):
        self.kgtype = config.kg_representation
        self.use_head = use_head
        self.config = config
        self.influential_only = influential_only
        self.data_provider = data_provider
        self.ig, self._changed_parameters = import_knowledge_graph(
            directory=self.config.embedding_folder,
            influential_only=influential_only,
            data_provider=data_provider,
            sdg_config=self.config.sdg_config,
            kg_config=self.config.aipe_kg_config,
            knowledge_extraction_method=embedding_config.knowledge_extraction_method,
            rule_extraction_method=embedding_config.rule_extraction_method,
            knowledge_extraction_weight_function=embedding_config.knowledge_extraction_weight_function,
            knowledge_extraction_filter_function=embedding_config.knowledge_extraction_filter_function
        )

        self._node_embeddings = NodeEmbeddings(
            base_folder=config.embedding_folder,
            influential_only=influential_only,
            use_head=use_head,
            embedding_type=EmbeddingType(config.embedding_method),
            kg_type=self.kgtype,
            random_seed=config.seed,
            with_test_data=test_data,
            embedding_config=embedding_config,
            changed_parameters=self._changed_parameters,
            graph=self.ig
        )
        self._embedding_dim = self._node_embeddings._embedding_dim
        if embedding_config:
            self._subkg_norm_method = embedding_config.subkg_normalization_method
        if generate_lut:
            self._skg_lut: DataFrame = self._load_lut(config.embedding_folder)

    @property
    def node_embeddings(self) -> NodeEmbeddings:
        return self._node_embeddings

    @property
    def edges(self) -> DataFrame:
        return self.ig.get_edge_dataframe()

    @property
    def vertecies(self) -> DataFrame:
        return self.ig.get_vertex_dataframe()

    def _generate_lut(self, file, file2):
        """Export the SKG-Embedding as a tsv-File

        Args:
            file (file): File to write the SKG-lut to
            file2 (file): File to write the SKG-Metadata to.

        Returns:
            _type_: _description_
        """

        # Differentiation between different KG-Types in order to call the skg
        # embedding correctly

        # In every possible if choice a v-stack of skg-embeddings is created
        # Every skg-embedding is started by propagating from a quality in the
        # graph. Therefore one skg-embedding for every quality is generated
        skg_lut = DataFrame(
            np.vstack([
                SubGraphEmbedding(
                    self.node_embeddings,
                    'jaccard',
                    False,
                    self.ig,
                    self.kgtype,
                    idx,
                    embedding_dim=self._embedding_dim,
                    vertices=self.node_embeddings.metadata,
                    norm_method=self._subkg_norm_method
                ).embedding.T
                for idx in self.vertecies.loc[
                    self.vertecies['type'] == 'qual_influence'
                ].index.values
            ])
        )

        with open(file, 'w', encoding='utf-8') as out_file:
            csv_str = skg_lut.to_csv(
                index=False, sep='\t', index_label=False, header=False)
            out_file.write(csv_str)

        skg_meta = DataFrame(
            np.vstack(
                self.vertecies.loc[self.vertecies['type']
                                   == 'qual_influence'].name
            )
        )
        with open(file2, 'w', encoding='utf-8') as out_file:
            csv_str = skg_meta.to_csv(
                index=False, sep='\t', index_label=False, header=False)
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
            return pd.read_csv(file, sep='\t', names=[i for i in range(0, self._embedding_dim)])
        else:
            return self._generate_lut(file, file2)

    def get_embedding_from_lut(self, rating, emb_dim) -> Tensor:
        try:
            indices: List[int] = self._get_indices_from_rating_names([rating])
            index_list = [
                idx for idx in
                self.vertecies.loc[
                    (self.vertecies['type'] == 'qual_influence') &
                    (self.vertecies['name'] != 'overall_ok')
                ].index.values
            ]
            emb = Tensor(self._skg_lut.loc[index_list.index(indices[0])])
        except IndexError:
            emb = zeros(emb_dim)
        return emb

    def get_sub_kg_from_ratings(
        self,
        ratings: List[str],
        distance_measure="euclidean",
        use_head=False
    ) -> Optional[
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

        skg_emb = SubGraphEmbedding(self.node_embeddings,  distance_measure, use_head,
                                    self.ig, self.kgtype, indices[0], self._embedding_dim, vertices=self.vertecies, norm_method=self._subkg_norm_method)

        return skg_emb

    def _get_indices_from_rating_names(self, ratings: List[str]) -> List[int]:
        """
        get indices of the rating names in the vertecies table
        :param ratings:
        :return:
        """
        return [self.vertecies.index[self.vertecies['name'] == rating][0] for rating in ratings]
