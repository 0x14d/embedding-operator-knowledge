from __future__ import annotations

from enum import Enum

class EmbeddingType(str, Enum):
    """
    This enum containts all available embedding methods
    """
    BoxE = "BoxE"
    ComplEx = "ComplEx"
    ComplExLiteral = "ComplExLiteral"
    DistMult = "DistMult"
    DistMultLiteralGated = "DistMultLiteralGated"
    HolE = "HolE"
    Random = "random"
    Rdf2Vec = "rdf2vec"
    Rescal = "Rescal"
    RotatE = "RotatE"
    SimplE = "SimplE"
    TorusE = "TorusE"
    TransE = "TransE"
    TransH = "TransH"
    TuckER = "TuckER"

    @property
    def latex_label(self) -> str:
        """Latex label of the embedding type"""
        labels = {
            EmbeddingType.ComplExLiteral: "ComplEx-LiteralE",
            EmbeddingType.DistMultLiteralGated : r"$\mathdefault{DistMult-LiteralE}_\mathdefault{g}$"
        }
        return labels.get(self, self.value)
