""" NodeEmbeddings class"""

# pylint: disable=import-error

import os
import sys
import multiprocessing as mp
from typing import Optional, Tuple

import igraph
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.optim import Adam
from torch import tensor  # pylint: disable=no-name-in-module
from torchkge import TransHModel
from torchkge.data_structures import KnowledgeGraph
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from tqdm.autonotebook import tqdm

from knowledge_infusion.graph_embeddings.embedding_types import EmbeddingType
from knowledge_infusion.graph_embeddings.embedding_config import EmbeddingConfig, NormalizationMethods, StandardConfig
from knowledge_extraction.rule_to_representation import *
from knowledge_infusion.graph_embeddings.utils.train_test_split import kg_train_test_split
from knowledge_infusion.rdf2vec.EvalDataset import EvalDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class NodeEmbeddings:
    """
    NodeEmbeddings class for calculating NodeEmbeddings from Knowledge Graph
    """
    _graph: igraph.Graph
    _torch_kge_graph: KnowledgeGraph
    _embeddings: DataFrame
    _edges: DataFrame
    _changed_parameters: DataFrame
    _use_head: bool
    _base_folder: str
    _embedding_type = EmbeddingType

    def __init__(
        self,
        base_folder: str,
        graph: igraph.Graph,
        influential_only=False,
        use_head=False,
        embedding_type: EmbeddingType = EmbeddingType.TransH,
        kg_type='basic',
        random_seed='1111',
        with_test_data=False,
        embedding_config: EmbeddingConfig = None,
        changed_parameters: Optional[DataFrame] = None
    ):
        # If no config is provided default to the standard config defined above
        if embedding_config is None:
            self.embedding_config = StandardConfig()
        else:
            self.embedding_config = embedding_config

        self._use_head = use_head
        self._influential_only = influential_only
        self._base_folder = base_folder
        self._embedding_type = embedding_type
        self._kgtype = kg_type
        self._random_seed = random_seed
        self._changed_parameters = changed_parameters
        self._graph = graph
        self._edges = graph.get_edge_dataframe()
        self._preprocess_kg_data()
        self._with_test_data = with_test_data
        self._embedding_dim = self.embedding_config.embedding_dim

        # Specify the number of epochs used for the different embedding types
        self._epochs = self.embedding_config.epochs

        if os.path.isfile(base_folder + 'graph_embeddings/node_embeddings.tsv'):
            self.import_tsv(base_folder + 'graph_embeddings/')
        else:
            self.train_embeddings_pykeen()

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    @property
    def metadata(self):
        return self._graph.get_vertex_dataframe()

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value):
        self._edges = value

    def get_embedding_and_metadata_by_idx(self, idx: int) -> Tuple[DataFrame, DataFrame]:
        """
        provides embedding and metadata by idx
        :param idx: identifier
        :return: embedding and metadata as pandas Dataframes
        """
        try:
            return self._embeddings.loc[idx], self.metadata.loc[idx]
        except KeyError:
            try:
                return self._embeddings.iloc[idx], self.metadata.loc[idx]
            except:
                print()

    def get_parameter_name_by_idx(self, idx: int) -> str:
        """
        get parameter name by idx
        :param idx: parameter id
        :return: parameter name
        """
        return self.metadata.loc[idx]

    def _preprocess_kg_data(self):
        """
         - replaces edge weights with mean parameter change over all experiments
         - initializes the torchkge KnowledgeGraph
        :return:
        """
        self._edges.rename(
            columns={'source': 'from', 'target': 'to', 'weight': 'rel'},
            inplace=True
        )
        self._torch_kge_graph = KnowledgeGraph(df=self._edges)

    def get_edge_weight_by_idx(
        self,
        edge_idx: int,
        source_idx: int,
        target_idx: int,
        edge_df: DataFrame,
        vertex_df: DataFrame
    ):
        """
        calculates mean and median from all parameter changes
        :param edge_idx:
        :param source_idx:
        :param target_idx:
        :param edge_df:
        :param vertex_df:
        :return: mean, median
        """
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
        """
        import embedding tsv files
        :param folder:
        :return:
        """
        emb_dim = self._embedding_dim
        self._embeddings = pd.read_csv(
            folder + 'node_embeddings.tsv',
            sep='\t',
            names=[i for i in range(0, emb_dim)]
        )

        # Since the Embedding Methods Complex(Literal) and RotatE are using imaginary Numbers
        # they are getting saved in the usual patter for those numbers: re + im*j
        # Pandas can save these numbers normally like every other number but cannot load them.
        # Thus we save these emebddings in an extra pickle file and load them from there
        # and not like the other methods were we just read in the csv.
        # If this pickle loading annoys you feel free to fix this issue in the pandas library
        # https://github.com/pandas-dev/pandas/issues/9379
        if self._embedding_type in \
           [EmbeddingType.ComplEx, EmbeddingType.ComplExLiteral, EmbeddingType.RotatE]:
            with open(folder + 'node_embeddings.pickle', 'rb') as out_file:
                self._embeddings = pickle.load(out_file)

    def _save_embeddings_and_metadata(self):
        """
        save embeddings as tsv files
        :return:
        """
        embeddings_file: str = '/node_embeddings.tsv'
        embeddings_metadata_file: str = '/node_embedding_metadata.tsv'
        embeddings_pik: str = '/node_embeddings.pickle'
        new_folder_name = self._base_folder + 'graph_embeddings/'

        with open(new_folder_name + embeddings_file, 'wt', encoding='utf-8') as out_file:
            self._embeddings = pd.DataFrame(self._embeddings[0].numpy())
            csv = self._embeddings.to_csv(
                index=False, sep='\t', index_label=False, header=False)
            out_file.write(csv)

        with open(new_folder_name + embeddings_metadata_file, 'wt', encoding='utf-8') as out_file:
            csv_meta_str = self.metadata.to_csv(
                index=False, index_label=False, sep='\t')
            out_file.write(csv_meta_str)

        if self._embedding_type in \
           [EmbeddingType.ComplEx, EmbeddingType.ComplExLiteral, EmbeddingType.RotatE]:
            with open(new_folder_name + embeddings_pik, 'wb') as out_file:
                pickle.dump(self._embeddings, out_file)

    def model_chooser(self, factory):
        """This sets up the model for the training depending on this objects
        property

        Args:
            factory (pykeen triples factory): -

        Returns:
            pykeen.models.*: The embedding model choosen when creating this
            object
        """
        if self._embedding_type == EmbeddingType.TransE:
            from pykeen.models import TransE
            model = TransE(
                triples_factory=factory,
                embedding_dim=int(self._embedding_dim),
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.ComplEx:
            from pykeen.models import ComplEx
            model = ComplEx(
                triples_factory=factory,
                embedding_dim=int(self._embedding_dim),
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.ComplExLiteral:
            from pykeen.models import ComplExLiteral
            model = ComplExLiteral(
                triples_factory=factory,
                embedding_dim=int(self._embedding_dim),
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.TransH:
            from pykeen.models import TransH
            model = TransH(
                triples_factory=factory,
                embedding_dim=int(self._embedding_dim),
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.DistMult:
            from pykeen.models import DistMult
            model = DistMult(
                triples_factory=factory,
                embedding_dim=int(self._embedding_dim),
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.RotatE:
            from pykeen.models import RotatE
            model = RotatE(
                triples_factory=factory,
                embedding_dim=int(self._embedding_dim),
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.BoxE:
            from pykeen.models import BoxE
            model = BoxE(
                triples_factory=factory,
                embedding_dim=int(self._embedding_dim),
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.DistMultLiteralGated:
            from pykeen.models import DistMultLiteralGated
            model = DistMultLiteralGated(
                triples_factory=factory,
                embedding_dim=self._embedding_dim,
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.TuckER:
            from pykeen.models import TuckER
            model = TuckER(
                triples_factory=factory,
                embedding_dim=self._embedding_dim,
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.HolE:
            from pykeen.models import HolE
            model = HolE(
                triples_factory=factory,
                embedding_dim=self._embedding_dim,
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.TorusE:
            from pykeen.models import TorusE
            model = TorusE(
                triples_factory=factory,
                embedding_dim=self._embedding_dim,
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.SimplE:
            from pykeen.models import SimplE
            model = SimplE(
                triples_factory=factory,
                embedding_dim=self._embedding_dim,
                random_seed=self._random_seed
            )

        elif self._embedding_type == EmbeddingType.Rescal:
            from pykeen.models import RESCAL
            model = RESCAL(
                triples_factory=factory,
                embedding_dim=self._embedding_dim,
                random_seed=self._random_seed
            )

        return model

    def train_embeddings(self):
        """
        Train Node embeddings with the torchkge framework
        :return:
        """
        # Define some hyper-parameters for training
        if self._use_head:
            emb_dim = 23
        else:
            emb_dim = 46
        lr = 0.0004
        n_epochs = 1000
        b_size = 4
        margin = 0.5

        # Define the model and criterion
        model = TransHModel(
            emb_dim, self._torch_kge_graph.n_ent, self._torch_kge_graph.n_rel)
        criterion = MarginLoss(margin)

        # Move everything to CUDA if available
        # if cuda is not None:
        #    if cuda.is_available():
        #       cuda.empty_cache()
        #      model.cuda()
        #     criterion.cuda()

        # Define the torch optimizer to be used
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        sampler = BernoulliNegativeSampler(self._torch_kge_graph)
        dataloader = DataLoader(self._torch_kge_graph, batch_size=b_size)

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

    def train_embeddings_pykeen(self):
        """Training of the embedding using a model from the pykeen library or
        from pyRDF2Vec
        """
        # Create a triples factory to feed the training pipeline
        factory, test_data = kg_train_test_split(
            kg_type=self._kgtype,
            edges=self.edges,
            metadata=self.metadata,
            seed=self._random_seed,
            test_split=self.embedding_config.train_test_split if self._with_test_data else 0.0,
            use_literals=self._embedding_type in [EmbeddingType.ComplExLiteral, EmbeddingType.DistMultLiteralGated]
        )
        if self._with_test_data:
            self.train_data, self.test_data = factory, test_data
            self.generate_graph_information()


        # Train random embeddings
        if self._embedding_type == EmbeddingType.Random:
            self.train_random_embeddings()
            return

        # Train Rdf2Vec
        if self._embedding_type == EmbeddingType.Rdf2Vec:
            self.train_rdf2vec()
            return

        # Choose the correct embedding method from pykeen
        model = self.model_chooser(factory)
        self.pykeen_model = model

        optimizer = Adam(params=model.parameters(),
                         lr=0.0004, weight_decay=1e-5)

        from pykeen.training import SLCWATrainingLoop

        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=factory,
            optimizer=optimizer,
            negative_sampler='bernoulli',

        )

        _ = training_loop.train(
            triples_factory=factory,
            num_epochs=self._epochs[self._embedding_type],
            drop_last=False,
        )

        from matplotlib import pyplot as plt
        plt.plot(_)
        plt.savefig(self._base_folder + 'lossplot.png')
        plt.clf()

        # Get the embeddings
        self.embeddings = [model.entity_representations[0]().detach()]
        self._save_embeddings_and_metadata()
        if self._with_test_data:
            to_save = {'model': model, 'test': self.test_data,
                       'train': self.train_data}
            model_save_data = self._base_folder + 'model.pickle'

            with open(model_save_data, 'wb') as out_f:
                pickle.dump(to_save, out_f)

    def prepare_rdf2vec_evaluation(self, data):
        """this method should evaluate the results from pyRDF2Vec embeddings
        with the following metrics. hits@, AMRI
        ! currently not implemented

        """
        pass

    def generate_graph_information(self, datatype="train"):
        """Generate informations about the graph to be able to describe it better

        Args:
            datatype (str, optional): _description_. Defaults to "train".
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from statistics import mean, stdev

        el = []
        for _, edge in self.edges.iterrows():
            if self._kgtype == "quantified_conditions_with_literal" and edge.loc['literal_included'] != 'None':
                continue
            elif self._kgtype == 'basic':
                el.append(str(edge.loc['from']) + "||" + str(edge.loc['to']))
            else:
                el.append(str(edge.loc['from']) + "||" + str(edge.loc['to']))

        G = nx.parse_edgelist(el, delimiter="||",
                              create_using=nx.DiGraph).to_directed()

        closness_centrality = nx.closeness_centrality(G).values()
        degree_centrality = nx.degree_centrality(G).values()
        avgnbdeg = nx.average_neighbor_degree(G, source="in+out").values()

        if "_with_literal" in self._kgtype:
            nolit = self.metadata.loc[self.metadata['type']
                                      == 'value'].shape[0]

        else:
            nolit = 0

        degrees = G.degree()

        sum_of_edges = sum(dict(degrees).values())
        avg_degree = sum_of_edges / G.number_of_nodes()

        graph_data = {
            'representation': self._kgtype,
            'No. of edges': len(el),
            'No. of nodes': len(self.metadata) - nolit,
            'No. of literals': nolit,
            'No. of relations': self.train_data.num_relations,
            'closeness centrality': str(round(mean(closness_centrality), 2)) + "±" + str(round(stdev(closness_centrality), 2)),
            'degree centrality': str(round(mean(degree_centrality), 2)) + "±" + str(round(stdev(degree_centrality), 2)),
            'avg. nb. degree': str(round(mean(avgnbdeg), 2)) + "±" + str(round(stdev(avgnbdeg), 2)),
            'avg. degree': avg_degree
        }
        with open(self._base_folder + self._kgtype + "_graphdata_" + datatype + ".pkl", 'wb') as out_f:
            pickle.dump(graph_data, out_f)

    def train_rdf2vec(self):
        """Train the embeddings using RDF2Vec
        """
        from pyrdf2vec.graphs import KG, Vertex
        from pyrdf2vec import RDF2VecTransformer
        from pyrdf2vec.embedders import Word2Vec
        from pyrdf2vec.walkers import RandomWalker

        os.environ['PYTHONHASHSEED'] = str(self._random_seed)

        # Preprocess data for pyrdf2vec
        URL = "http://pyRDF2Vec"
        
        preprocessed_train_data = []
        for relation in self.train_data.triples:
            preprocessed_train_data.append([
                f"{URL}#{relation[0]}".replace(" ", '_'),
                f"{URL}#{relation[1]}".replace(" ", "_"),
                f"{URL}#{relation[2]}".replace(" ", "_"),
            ])
            
        preprocessed_test_data = []
        for relation in self.test_data.triples:
            preprocessed_test_data.append([
                f"{URL}#{relation[0]}".replace(" ", '_'),
                f"{URL}#{relation[1]}".replace(" ", "_"),
                f"{URL}#{relation[2]}".replace(" ", "_"),
            ])
        
        # create an RDF2Vec specific grpah object 
        CUSTOM_KG = KG()
        for row in preprocessed_train_data:
            subj = Vertex(row[0])
            obj = Vertex(row[2])
            pred = Vertex(row[1], predicate=True, vprev=subj, vnext=obj)
            CUSTOM_KG.add_walk(subj, pred, obj)

        # Create entity list of all entities (from train and test set)
        entities = []
        for index, row in self.metadata.iterrows():
            vert_str = "http://pyRDF2Vec#" + str(self.metadata.loc[index]['name']).replace(" ", "_")
            entities.append(vert_str)
            CUSTOM_KG.add_vertex(Vertex(vert_str))
            
        transformer = RDF2VecTransformer(
            Word2Vec(epochs=self._epochs[self._embedding_type],
                    vector_size=int(self._embedding_dim),
                    workers=1),
            walkers=[RandomWalker(self.embedding_config.rdf2vec_walker_max_depth, self.embedding_config.rdf2vec_walker_max_walks, with_reverse=False, n_jobs=2, random_state=self._random_seed)],
            )
        
        # Do the embedding in another thread in order for change to PYTHONHASHSEED to work correctly
        q = mp.Queue()
        p = mp.Process(target=fit_transform_wrapper, args=(transformer, CUSTOM_KG, entities, q,))
        p.start()
        embeddings, literals, model = q.get()
        p.join()
        
        
        # Save the gensim model for the kbc_evaluation
        model.save(self._base_folder + "model.model")
        # Save the train data as an nt file for the kbc_evaluation
        self.save_dataset_as_nt_file(preprocessed_train_data)
        # Save the whole dataset as a
        eval_data = EvalDataset(preprocessed_train_data, preprocessed_test_data)
        
        with open(self._base_folder + "eval_dataset_object.pickle", "wb") as out_f:
            pickle.dump(eval_data, out_f)
        
        # Save the emebeddings in the usual f
        self.embeddings = [tensor(embeddings)]

        to_save = {'embeddings': self.embeddings, 'test' : preprocessed_test_data, 'train' : preprocessed_train_data}
        model_save_data = self._base_folder + 'model.pickle'

        with open(model_save_data, 'wb') as out_f:
            pickle.dump(to_save, out_f)
        self._save_embeddings_and_metadata()

    def train_random_embeddings(self):
        rng = np.random.default_rng(seed=self._random_seed)
        random_embeddings = [
            [rng.uniform(-1, 1) for _ in range(self._embedding_dim)] for _ in range(len(self.metadata))
        ]
        self._embeddings = tensor([random_embeddings])
        self._save_embeddings_and_metadata()
        
    def save_dataset_as_nt_file(self, train_data):
        """Save the dataset in the nt file format.
        """
        with open(self._base_folder + "dataset.nt", "w+", encoding="utf8") as f:
            for triple in train_data:
                f.write(
                    "<" + triple[0] + "> <" + triple[1] + "> <" + triple[2] + "> .\n"
                )


def fit_transform_wrapper(transformer, kg, ents, q):
    """Wrapper for the training of rdf2vec.
    Since this is an extra method the training can be called in seperate thread
    which leads to the PYTHONHASHSEED being manipulable and thus ensuring proper
    seeding is possible

    Args:
        transformer : the pyRDF2Vec transformer
        kg : The pyRDF2Vec Knowledge Graph
        ents (List): List of all entities
        q (Queue): Queue for the results
    """

    embeddings, literals = transformer.fit_transform(kg, ents)
    model = transformer.embedder._model
    q.put((embeddings, literals, model))
    return
