class MatchesAtKResult:
    embedding: str
    representation: str
    distance_measure: str
    use_head: bool
    mean: float
    std: float
    k: int

    def __init__(self, embedding, representation, distance_measure, use_head, mean, std, k):
        self.embedding = embedding
        self.representation = representation
        self.distance_measure = distance_measure
        self.use_head = use_head
        self.mean = mean
        self.std = std
        self.k = k