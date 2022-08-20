""" NodeEmbeddings class"""
from html import entities
import os
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import cuda
from torch.optim import Adam
from torch import tensor
# from torch.version import cuda
from torchkge import TransHModel
from torchkge.data_structures import KnowledgeGraph
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from tqdm.autonotebook import tqdm
import igraph
import pickle


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.abstract_knowledge_graph_generator import KnowledgeGraphGenerator
from knowledge_extraction.graph_aggregation import GraphAggregation, Graph
import experiment_definition as ed
import knowledge_graph as kg


class NodeEmbeddings:
    """
    NodeEmbeddings class for calculating NodeEmbeddings from Knowledge Graph
    """
    _graph: Graph
    _igraph_graph: igraph.Graph
    _knowledge_graph: KnowledgeGraph
    _embeddings: DataFrame
    _metadata: DataFrame
    _edges: DataFrame
    _changed_parameters: DataFrame
    _use_head: bool
    _base_folder: str
    _type = str
    _knowledge_graph_generator= KnowledgeGraphGenerator

    def __init__(self, base_folder: str, node_embeddings=None, influential_only=False, use_head=False, type="TransH", kg_type='basic', random_seed='1111', knowledge_graph_generator=None):
        self._use_head = use_head
        self._influential_only = influential_only
        self._base_folder = base_folder
        self._type = type
        self._kgtype = kg_type
        self._random_seed = random_seed
        self._knowledge_graph_generator = knowledge_graph_generator
        self._import_knowledge_graph(base_folder)
        self._preprocess_kg_data()
        if node_embeddings is None:
            if os.path.isfile(base_folder + 'graph_embeddings/node_embeddings.tsv'):
               self.import_tsv(base_folder + 'graph_embeddings/')
            else:
                self.train_embeddings_pykeen()
                #self.train_embeddings()
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
        """
        provides embedding and metadata by idx
        :param idx: identifier
        :return: embedding and metadata as pandas Dataframes
        """
        try: 
            return self._embeddings.loc[idx], self._metadata.loc[idx]
        except KeyError:
            return self._embeddings.iloc[idx], self._metadata.loc[idx]

    def get_parameter_name_by_idx(self, idx: int) -> str:
        """
        get parameter name by idx
        :param idx: parameter id
        :return: parameter name
        """
        return self._metadata.loc[idx]

    def _get_graph_from_database(self):
        """
        provides Data from MongoDB
        :return:
        """

        self._graph = self._knowledge_graph_generator.generate_knowledge_graph()
        self._changed_parameters = None
        self._metadata = self._graph.get_vertex_dataframe()
        self._edges = self._graph.get_edge_dataframe()

    def _save_graph_data(self, folder=''):
        """
        saves graph data as pkl files
        :param folder:
        :return:
        """
        df_ed = self._graph.get_edge_dataframe()
        df_vs = self._graph.get_vertex_dataframe()
        if self._influential_only:
            file_name_add = '_inf_only.pkl'
        else:
            file_name_add = '.pkl'
        if not os.path.isdir(folder + 'graph_embeddings/'):
            os.makedirs(folder + 'graph_embeddings/')
        if not os.path.exists(folder + 'graph_embeddings/edges' + file_name_add):
            df_ed.to_pickle(folder + 'graph_embeddings/edges' + file_name_add)
        if not os.path.exists(folder + 'graph_embeddings/verts' + file_name_add):
            df_vs.to_pickle(folder + 'graph_embeddings/verts' + file_name_add)
        if self._changed_parameters is not None and not os.path.exists(
                folder + 'graph_embeddings/parameters' + file_name_add):
            self._changed_parameters.to_pickle(folder + 'graph_embeddings/parameters' + file_name_add)

    def _import_knowledge_graph(self, folder):
        """
        imports graph pkl files
        :param folder:
        :return:
        """
        if self._influential_only:
            file_name_add = '_inf_only.pkl'
        else:
            file_name_add = '.pkl'
        try:
            self._edges = pd.read_pickle(folder + 'graph_embeddings/edges' + file_name_add)
            self._metadata = pd.read_pickle(folder + 'graph_embeddings/verts' + file_name_add)
        except (FileNotFoundError, ValueError):
            print("Import Graph Data from Database")
            self._get_graph_from_database()
            self._save_graph_data(folder)

    def _preprocess_kg_data(self):
        """
         - replaces edge weights with mean parameter change over all experiments
         - initializes the torchkge KnowledgeGraph
        :return:
        """
        self._edges.rename(columns={'source': 'from', 'target': 'to', 'weight': 'rel'}, inplace=True)
        self._knowledge_graph = KnowledgeGraph(df=self._edges)

    def get_edge_weight_by_idx(self, edge_idx: int, source_idx: int, target_idx: int, edge_df: DataFrame,
                               vertex_df: DataFrame):
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
        if self._use_head:
            emb_dim = 48
        else:
            emb_dim = 48
        self._embeddings = pd.read_csv(folder + 'node_embeddings.tsv', sep='\t', names=[i for i in range(0, emb_dim)])
        self._metadata = pd.read_csv(folder + 'node_embedding_metadata.tsv', sep='\t')

        if self._type == 'ComplEx' or self._type == 'ComplExLiteral' or self._type == 'RotatE':
            with open(folder + 'node_embeddings.pickle', 'rb') as out_file:
                self._embeddings = pickle.load(out_file)
        pass

    def _save_embeddings_and_metadata(self):
        """
        save embeddings as tsv files
        :return:
        """
        embeddings_file: str = '/node_embeddings.tsv'
        embeddings_metadata_file: str = '/node_embedding_metadata.tsv'
        embeddings_pik: str = '/node_embeddings.pickle'
        # version_folder: str = 'version'
        # current_version: int = 1
        # while os.path.isdir(version_folder + str(current_version)):
        #    current_version = current_version + 1

        # new_folder_name: str = version_folder + str(current_version)
        # os.mkdir(new_folder_name)
        new_folder_name = self._base_folder + 'graph_embeddings/'

        with open(new_folder_name + embeddings_file, 'wt') as out_file:
            self._embeddings = pd.DataFrame(self._embeddings[0].numpy())
            csv_str = self._embeddings.to_csv(index=False, sep='\t', index_label=False, header=False)
            out_file.write(csv_str)

        with open(new_folder_name + embeddings_metadata_file, 'wt') as out_file:
            csv_meta_str = self._metadata.to_csv(index=False, index_label=False, sep='\t')
            out_file.write(csv_meta_str)

        if self._type == 'ComplEx' or self._type == 'ComplExLiteral' or self._type == 'RotatE':
            with open(new_folder_name + embeddings_pik, 'wb') as out_file:
                pickle.dump(self._embeddings, out_file)


    def split_off_literals(self):
        """
        Splits the relations of the graph into 3xn-Matrix containing all unquantified relations and 
        one 3xm-Matrix containing the quantifying relations with literals. This is needed for the 
        Embedding Methods, which take literals into account.
        """
        
        literals = [] # TODO Rename literal_relations | entity_relations
        relations = []
        if self._kgtype == 'basic':
            for _, row in self.edges.iterrows():
                source_int = row.loc['from']
                to_int = row.loc['to']
                rel_float = row.loc['rel']
                source_name = self._metadata.loc[source_int]['name']
                to_name = self.metadata.loc[to_int]['name']
                relations.append([source_name, str(rel_float), to_name])

        elif self._kgtype == 'unquantified':
            for _, row in self.edges.iterrows():
                source_int = row.loc['from']
                to_int = row.loc['to']
                rel_float = row.loc['rel']
                source_name = self._metadata.loc[source_int]['name']
                to_name = self.metadata.loc[to_int]['name']
                relations.append([source_name, "implies", to_name])

        elif self._kgtype == 'quantified_parameters_with_shortcut' or self._kgtype == 'quantified_parameters_without_shortcut':
            for _, row in self.edges.iterrows():
                lit = row.loc['literal_included']
                source_int = row.loc['from']
                to_int = row.loc['to']
                rel_float = row.loc['rel']
                source_name = self._metadata.loc[source_int]['name']
                to_name = self.metadata.loc[to_int]['name']
                if row.loc['literal_included'] == 'From':
                    relations.append([source_name, "quantifies", to_name])
                elif row.loc['literal_included'] == 'To':
                    relations.append([source_name, "implies", to_name])
                elif row.loc['literal_included'] == "None":
                    relations.append([source_name, "implies", to_name])

        elif self._kgtype == 'quantified_parameters_with_literal':
            for _, row in self.edges.iterrows():
                lit = row.loc['literal_included']
                source_int = row.loc['from']
                to_int = row.loc['to']
                rel_float = row.loc['rel']
                source_name = self._metadata.loc[source_int]['name']
                to_name = self.metadata.loc[to_int]['name']
                if row.loc['literal_included'] == 'From': # TODO comment umdrehen der relation why
                    literals.append([to_name, "quantified by", source_name])
                elif row.loc['literal_included'] == 'To':
                    literals.append([source_name, "implies", to_name])
                elif row.loc['literal_included'] == "None":
                    relations.append([source_name, "implies", to_name])
        
        elif self._kgtype == 'quantified_conditions':
            for _, row in self.edges.iterrows:
                raise NotImplementedError()

        np_rel = np.array(relations)
        np_lit = np.array(literals)

        if self._kgtype == 'unquantified' or self._kgtype=='basic' or self._kgtype== 'quantified_parameters_with_shortcut' or self._kgtype == 'quantified_parameters_without_shortcut':
            np_lit = np.array([[0,0,0]])
        return (np_rel, np_lit)

    def model_chooser(self, factory):
        if self._use_head:
            k = 1
        else:
            k = 1
        
        if self._type == "TransE":
            from pykeen.models import TransE
            model=TransE(
                triples_factory=factory,
                embedding_dim=int(48*k),
                random_seed=self._random_seed
                )

        elif self._type == "ComplEx":
            from pykeen.models import ComplEx
            model=ComplEx(
                triples_factory=factory,
                embedding_dim=int(48*k),
                random_seed=self._random_seed
            )

        elif self._type == "ComplExLiteral":
            self.split_off_literals()
            from pykeen.models import ComplExLiteral
            model=ComplExLiteral(
                triples_factory=factory,
                embedding_dim=int(48*k),
                random_seed=self._random_seed
            )
        
        elif self._type == "TransH":
            from pykeen.models import TransH
            model=TransH(
                triples_factory=factory,
                embedding_dim=int(48*k),
                random_seed=self._random_seed
            )

        elif self._type == "DistMult":
            from pykeen.models import DistMult
            model=DistMult(
                triples_factory=factory,
                embedding_dim=int(48*k),
                random_seed=self._random_seed
            )

        elif self._type == "RotatE":
            from pykeen.models import RotatE
            model=RotatE(
                triples_factory=factory,
                embedding_dim=int(48*k),
                random_seed=self._random_seed
            )

        elif self._type == "BoxE":
            from pykeen.models import BoxE
            model = BoxE(
                triples_factory=factory,
                embedding_dim=int(48*k),
                random_seed=self._random_seed
            )
        
        elif self._type == "DistMultLiteralGated":
            from pykeen.models import DistMultLiteralGated
            model=DistMultLiteralGated(
                triples_factory=factory,
                embedding_dim=48,
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
        model = TransHModel(emb_dim, self._knowledge_graph.n_ent, self._knowledge_graph.n_rel)
        criterion = MarginLoss(margin)

        # Move everything to CUDA if available
        #if cuda is not None:
        #    if cuda.is_available():
         #       cuda.empty_cache()
          #      model.cuda()
           #     criterion.cuda()

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

    def train_embeddings_pykeen(self):

        if self._type == "rdf2vec":
            self.train_rdf2vec()
            return

        mapping = {}
        for index, row in self.metadata.iterrows():
            # Do not add literals to the mapping
            if self._kgtype == 'quantified_parameters_with_literal' and self.metadata.loc[index]['type'] is None:
                continue
            mapping[str(self.metadata.loc[index]['name'])] = index

        # Create a triples factory to feed the training pipeline
        if self._type != 'ComplExLiteral' and self._type != "DistMultLiteralGated":
            relations, literals = self.split_off_literals()
            from pykeen.triples import TriplesFactory
            factory = TriplesFactory.from_labeled_triples(relations, entity_to_id=mapping)
        
        else:
            from pykeen.triples import TriplesNumericLiteralsFactory
            relations, literals = self.split_off_literals()
            factory = TriplesNumericLiteralsFactory.from_labeled_triples(triples=relations, numeric_triples=literals, entity_to_id=mapping)
            
        
        # Choose the correct embedding method from pykeen
        model = self.model_chooser(factory)

        optimizer = Adam(params=model.parameters(), lr=0.0004, weight_decay=1e-5)

        from pykeen.training import SLCWATrainingLoop

        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=factory,
            optimizer=optimizer,
            negative_sampler='bernoulli'
        )

        _ = training_loop.train(
            triples_factory=factory,
            num_epochs=750,
        )

        # Get the embeddings
        self.embeddings = [model.entity_representations[0]().detach()]
        self._save_embeddings_and_metadata()

    def train_rdf2vec(self):
        from pyrdf2vec.graphs import KG, Vertex
        from pyrdf2vec import RDF2VecTransformer
        from pyrdf2vec.embedders import Word2Vec
        from pyrdf2vec.walkers import RandomWalker

        os.environ['PYTHONHASHSEED'] = str(self._random_seed)
        print("env set")
        if self._use_head:
            k = 1
        else:
            k = 1

        # Prepare data for pyrdf2vec
        relations, literals = self.split_off_literals()
        
        URL = "http://pyRDF2Vec"
        CUSTOM_KG = KG()
        for row in relations:
            subj = Vertex(f"{URL}#{row[0]}")
            obj = Vertex((f"{URL}#{row[2]}"))
            pred = Vertex((f"{URL}#{row[1]}"), predicate=True, vprev=subj, vnext=obj)
            CUSTOM_KG.add_walk(subj, pred, obj)

        if self._kgtype == 'quantified_parameters_with_literal':
            for row in literals:
                subj = Vertex(f"{URL}#{row[0]}")
                obj = Vertex((f"{URL}#{row[2]}"))
                pred = Vertex((f"{URL}#{row[1]}"), predicate=True, vprev=subj, vnext=obj)
                CUSTOM_KG.add_walk(subj, pred, obj)

        entities = []
        for index, row in self.metadata.iterrows():
            entities.append("http://pyRDF2Vec#" + str(self.metadata.loc[index]['name']))

        transformer = RDF2VecTransformer(
            Word2Vec(epochs=750,
                    vector_size=int(48*k),
                    workers=1),
            walkers=[RandomWalker(4, 10, with_reverse=False, n_jobs=2, random_state=self._random_seed)],
            # verbose=1
        )
        # Get our embeddings.
        q = mp.Queue()
        try:
            mp.set_start_method('spawn')
        except:
            print("err")
        # Do the embedding in another thread in order for change to PYTHONHASHSEED to work correctly
        p = mp.Process(target=fit_transform_wrapper, args=(transformer, CUSTOM_KG, entities, q,))
        p.start()
        embeddings, literal = q.get()
        p.join()
        self.embeddings = [tensor(embeddings)]
        self._save_embeddings_and_metadata()


def fit_transform_wrapper(transformer, kg, ents, q):
    print("inside")
    embeddings, literals = transformer.fit_transform(kg, ents)
    print("before put")
    q.put((embeddings, literals))
    print("after put")
    return


