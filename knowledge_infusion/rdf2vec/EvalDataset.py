class EvalDataset:
    """Since kbc_rdf2vec s dataset class is not expandable, this class can be used
    as a drop-in replacement due to pythons duck-typing
    """
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        
    def train_set(self):
        return self.train_data
    
    def test_set(self):
        return self.test_data
    
    def valid_set(self):
        return [["","",""]]