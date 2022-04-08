""" NodeEmbeddings class"""
import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.optim import Adam
from torch.version import cuda
from torchkge import TransHModel
from torchkge.data_structures import KnowledgeGraph
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from tqdm.autonotebook import tqdm


class NodeEmbeddings:
    """
    NodeEmbeddings class for calculating NodeEmbeddings from Knowledge Graph
    """
    _knowledge_graph: DataFrame
    _embeddings: DataFrame
    _metadata: DataFrame
    _edges: DataFrame
    _changed_parameters: DataFrame
    _use_head: bool

    def __init__(self, base_folder: str, node_embeddings=None, influential_only=False, use_head=False):
        self._use_head = use_head
        self._influential_only = influential_only
        self._import_knowledge_graph(base_folder)
        self._preprocess_kg_data()
        if node_embeddings is None:
            self.train_embeddings()
        else:
            self.import_tsv(node_embeddings)

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value):
        self._edges = value

    def get_embedding_and_metadata_by_idx(self, idx: int) -> (DataFrame, DataFrame):
        return self._embeddings.loc[idx], self._metadata.loc[idx]

    def get_parameter_name_by_idx(self, idx: int) -> str:
        return self._metadata.loc[idx]

    def _import_knowledge_graph(self, folder='graph_data/'):
        if self._influential_only:
            file_name_add = '_inf_only.pkl'
        else:
            file_name_add = '.pkl'
        try:
            self._edges = pd.read_pickle()
        except (FileNotFoundError):
            print("File not found: " + folder + 'graph_embeddings/edges' + file_name_add)
        try:
            self._metadata = pd.read_pickle(folder + 'graph_embeddings/verts' + file_name_add)
        except (FileNotFoundError):
            print("File not found: " + folder + 'graph_embeddings/verts' + file_name_add)
        try:
            self._changed_parameters = pd.read_pickle(folder + 'graph_embeddings/parameters' + file_name_add)
        except (FileNotFoundError):
            print("File not found: " + folder + 'graph_embeddings/parameters' + file_name_add)

    def _preprocess_kg_data(self):
        # adjust edge weights
        # mean/median over all weights
        for i, row in self._edges.iterrows():
            source_idx = self._edges.loc[i]['source']
            target_idx = self._edges.loc[i]['target']
            mean, _ = self.get_edge_weight_by_idx(i, source_idx, target_idx, self._edges, self._metadata)
            self._edges.at[i, 'weight'] = mean
        self._edges.drop('experiments', axis=1, inplace=True)
        self._edges.rename(columns={'source': 'from', 'target': 'to', 'weight': 'rel'}, inplace=True)

        self._metadata.drop(columns=["user_value", "original_value", "value"], inplace=True)

        self._knowledge_graph = KnowledgeGraph(df=self._edges)

    def get_edge_weight_by_idx(self, edge_idx: int, source_idx: int, target_idx: int, edge_df: DataFrame,
                               vertex_df: DataFrame):
        experiment_idx_list = edge_df.loc[edge_idx]['experiments']
        material_exps = list(vertex_df.loc[source_idx]['value'].keys())
        parameter_key = vertex_df.loc[target_idx]['key']
        changed_parameters = [self._changed_parameters.loc[i][parameter_key]
                              for i in self._changed_parameters.index.values
                              if i in set(experiment_idx_list).intersection(material_exps)]
        mean = np.mean(changed_parameters)
        median = np.median(changed_parameters)
        return mean, median

    def import_tsv(self, folder: str):
        if self._use_head:
            emb_dim = 23
        else:
            emb_dim = 46
        self._embeddings = pd.read_csv(folder + 'node_embeddings.tsv', sep='\t', names=[i for i in range(0, emb_dim)])
        self._metadata = pd.read_csv(folder + 'node_embedding_metadata.tsv', sep='\t')

    def _save_embeddings_and_metadata(self):

        embeddings_file: str = '/node_embeddings.tsv'
        embeddings_metadata_file: str = '/node_embedding_metadata.tsv'
        version_folder: str = 'version'
        current_version: int = 1
        while os.path.isdir(version_folder + str(current_version)):
            current_version = current_version + 1

        new_folder_name: str = version_folder + str(current_version)
        os.mkdir(new_folder_name)

        with open(new_folder_name + embeddings_file, 'wt') as out_file:
            csv_str = pd.DataFrame(self._embeddings[0].numpy()).to_csv(index=False, sep='\t', index_label=False,
                                                                       header=False)
            out_file.write(csv_str)

        with open(new_folder_name + embeddings_metadata_file, 'wt') as out_file:
            csv_meta_str = self._metadata.to_csv(index=False, index_label=False, sep='\t')
            out_file.write(csv_meta_str)

    def train_embeddings(self):
        # Define some hyper-parameters for training
        emb_dim = 23
        lr = 0.0004
        n_epochs = 1000
        b_size = 4
        margin = 0.5

        # Define the model and criterion
        model = TransHModel(emb_dim, self._knowledge_graph.n_ent, self._knowledge_graph.n_rel)
        criterion = MarginLoss(margin)

        # Move everything to CUDA if available
        if cuda is not None:
            if cuda.is_available():
                cuda.empty_cache()
                model.cuda()
                criterion.cuda()

        # Define the torch optimizer to be used
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        sampler = BernoulliNegativeSampler(self._knowledge_graph)
        dataloader = DataLoader(self._knowledge_graph, batch_size=b_size)

        iterator = tqdm(range(n_epochs), unit='epoch')
        for epoch in iterator:
            running_loss = 0.0
            for i, batch in enumerate(dataloader):
                h, t, r = batch[0], batch[1], batch[2]
                n_h, n_t = sampler.corrupt_batch(h, t, r)

                optimizer.zero_grad()

                # forward + backward + optimize
                pos, neg = model(h, t, n_h, n_t, r)
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                      running_loss / len(dataloader)))

        model.normalize_parameters()
        self._embeddings = model.get_embeddings()
        self._save_embeddings_and_metadata()
