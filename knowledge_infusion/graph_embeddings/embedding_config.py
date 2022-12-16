from abc import ABC
from enum import Enum
from knowledge_infusion.graph_embeddings.embedding_types import EmbeddingType

class NormalizationMethods(str, Enum):
    unit_norm = "unit_norm"
    number_nodes = "num_nodes"
    none = "none"


class EmbeddingConfig(ABC):
    """This class can be given to the Embedding class to configure non standard
    behaviour. 
    """
    epochs: dict
    embedding_dim: int

    train_test_split: float

    rdf2vec_walker_max_depth: int
    rdf2vec_walker_max_walks: int

    subkg_normalization_method: NormalizationMethods

    def __init__(self, epochs: dict, embedding_dim: int,
                 train_test_split: float, rdf2vec_walker_max_depth: int,
                 rdf2vec_walker_max_walks: int, subkg_normalization_method: NormalizationMethods) -> None:
        super().__init__()

        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.train_test_split = train_test_split
        self.rdf2vec_walker_max_depth = rdf2vec_walker_max_depth
        self.rdf2vec_walker_max_walks = rdf2vec_walker_max_walks
        self.subkg_normalization_method = subkg_normalization_method
        
class StandardConfig(EmbeddingConfig):
    def __init__(self) -> None:
        super().__init__(
            epochs={
                EmbeddingType.TransE: 400,
                EmbeddingType.ComplEx: 1000,
                EmbeddingType.ComplExLiteral: 650,
                EmbeddingType.RotatE: 700,
                EmbeddingType.DistMult: 800,
                EmbeddingType.DistMultLiteralGated: 200,
                EmbeddingType.BoxE: 1500,
                EmbeddingType.Rdf2Vec: 1000
            },
            embedding_dim=46,
            train_test_split=0.2,
            rdf2vec_walker_max_depth=4,
            rdf2vec_walker_max_walks=2, 
            subkg_normalization_method=NormalizationMethods.number_nodes
        )
