import unittest
from knowledge_infusion.graph_embeddings.node_embeddings import NodeEmbeddings
import pandas as pd
import numpy as np

class TestEdgesToEmbedding(unittest.TestCase):
    """
    This TestCase tests the method NodeEmbeddings.split_off_literals

    Test data is of the form:
    q1 -> p1
    q1 -> p2
    q2 -> p2
    adopted to fit given representation
    """

    def setUp(self):

        self.node_embedding = NodeEmbeddings.__new__(NodeEmbeddings)


    def test_unquantified(self):
        """
        Tests if literals and relations are correctly split off and build for unquantified representation
        """

        # Built the knowledge graph
        self.node_embedding.edges = pd.DataFrame(
            [
                [2, 0, 'implies'],
                [2, 1, 'implies'],
                [3, 1, 'implies'],
    	    ],
            columns=['from','to', 'rel']
        )
        self.node_embedding.metadata = pd.DataFrame(
            [
                ['param1', 'parameter'],
                ['param2', 'parameter'],
                ['qual1', 'qual_influence'],
                ['qual2', 'qual_influence'],
            ],
            columns=['name', 'type']
        )
        self.node_embedding._kgtype = 'unquantified'

        # Execute the method
        relations, literal = self.node_embedding.split_off_literals()

        # Built the expected results
        expected_relations = np.array([
                ['qual1', 'implies', 'param1'],
                ['qual1', 'implies', 'param2'],
                ['qual2', 'implies', 'param2']
            ])

        expected_literals = np.array([[0,0,0]])

        # Assert reality meets expectations
        self.assertTrue(np.array_equal(expected_literals, literal))
        self.assertTrue(np.array_equal(expected_relations, relations))


    def test_basic(self):
        # Build the input data
        self.node_embedding.edges = pd.DataFrame(
            [
                [2, 0, -1.0],
                [2, 1, -1.0],
                [3, 1, -1.0],
    	    ],
            columns=['from','to', 'rel']
        )
        self.node_embedding.metadata = pd.DataFrame(
            [
                ['param1', 'parameter'],
                ['param2', 'parameter'],
                ['qual1', 'qual_influence'],
                ['qual2', 'qual_influence'],
            ],
            columns=['name', 'type']
        )

        self.node_embedding._kgtype = 'basic'

        # Execute the method
        relations, literal = self.node_embedding.split_off_literals()

        # Built the expected results
        expected_relations = np.array([
                ['qual1', '-1.0', 'param1'],
                ['qual1', '-1.0', 'param2'],
                ['qual2', '-1.0', 'param2']
            ])

        expected_literals = np.array([[0,0,0]])

        # Assert reality meets expectations
        self.assertTrue(np.array_equal(expected_literals, literal))
        self.assertTrue(np.array_equal(expected_relations, relations))


    def test_with_shortcut(self):
        self.node_embedding.edges = pd.DataFrame(
            [
                [2, 4, 'implies', 'To'],
                [4, 0, 'quantifies', 'From'],
                [2, 0, 'implies', 'None'],
                [2, 5, 'implies', 'To'],
                [5, 1, 'quantifies', 'From'],
                [2, 1, 'implies', 'None'],
                [3, 6, 'implies', 'To'],
                [6, 1, 'quantifies', 'From'],
                [3, 1, 'implies', 'None'],
    	    ],
            columns=['from','to', 'rel', 'literal_included']
        )
        self.node_embedding.metadata = pd.DataFrame(
            [
                ['param1', 'parameter'],
                ['param2', 'parameter'],
                ['qual1', 'qual_influence'],
                ['qual2', 'qual_influence'],
                [-1.0, None],
                [-1.0, None],
                [-1.0, None],
            ],
            columns=['name', 'type']
        )

        self.node_embedding._kgtype = 'quantified_parameters_with_shortcut'

        # Execute the method
        relations, literal = self.node_embedding.split_off_literals()

        # Built the expected results
        expected_relations = np.array([
                ['qual1', 'implies', '-1.0'],
                ['-1.0', 'quantifies', 'param1'],
                ['qual1', 'implies', 'param1'],
                
                ['qual1', 'implies', '-1.0'],
                ['-1.0', 'quantifies', 'param2'],
                ['qual1', 'implies', 'param2'],

                ['qual2', 'implies', '-1.0'],
                ['-1.0', 'quantifies', 'param2'],
                ['qual2', 'implies', 'param2'],
            ])

        expected_literals = np.array([[0,0,0]])

        # Assert reality meets expectations
        self.assertTrue(np.array_equal(expected_literals, literal))
        self.assertTrue(np.array_equal(expected_relations, relations))

    def test_without_shortcut(self):
        self.node_embedding.edges = pd.DataFrame(
            [
                [2, 4, 'implies', 'To'],
                [4, 0, 'quantifies', 'From'],
                [2, 5, 'implies', 'To'],
                [5, 1, 'quantifies', 'From'],
                [3, 6, 'implies', 'To'],
                [6, 1, 'quantifies', 'From'],
    	    ],
            columns=['from','to', 'rel', 'literal_included']
        )
        self.node_embedding.metadata = pd.DataFrame(
            [
                ['param1', 'parameter'],
                ['param2', 'parameter'],
                ['qual1', 'qual_influence'],
                ['qual2', 'qual_influence'],
                [-1.0, None],
                [-1.0, None],
                [-1.0, None],
            ],
            columns=['name', 'type']
        )

        self.node_embedding._kgtype = 'quantified_parameters_with_shortcut'

        # Execute the method
        relations, literal = self.node_embedding.split_off_literals()

        # Built the expected results
        expected_relations = np.array([
                ['qual1', 'implies', '-1.0'],
                ['-1.0', 'quantifies', 'param1'],
                
                ['qual1', 'implies', '-1.0'],
                ['-1.0', 'quantifies', 'param2'],

                ['qual2', 'implies', '-1.0'],
                ['-1.0', 'quantifies', 'param2'],
            ])

        expected_literals = np.array([[0,0,0]])

        # Assert reality meets expectations
        self.assertTrue(np.array_equal(expected_literals, literal))
        self.assertTrue(np.array_equal(expected_relations, relations))

    def test_with_literal(self):
        self.node_embedding.edges = pd.DataFrame(
            [
                [2, 4, 'implies', 'To'],
                [4, 0, 'quantifies', 'From'],
                [2, 0, 'implies', 'None'],
                [2, 5, 'implies', 'To'],
                [5, 1, 'quantifies', 'From'],
                [2, 1, 'implies', 'None'],
                [3, 6, 'implies', 'To'],
                [6, 1, 'quantifies', 'From'],
                [3, 1, 'implies', 'None'],
    	    ],
            columns=['from','to', 'rel', 'literal_included']
        )
        self.node_embedding.metadata = pd.DataFrame(
            [
                ['param1', 'parameter'],
                ['param2', 'parameter'],
                ['qual1', 'qual_influence'],
                ['qual2', 'qual_influence'],
                [-1.0, None],
                [-1.0, None],
                [-1.0, None],
            ],
            columns=['name', 'type']
        )

        self.node_embedding._kgtype = 'quantified_parameters_with_literal'

        # Execute the method
        relations, literal = self.node_embedding.split_off_literals()

        # Built the expected results
        expected_relations = np.array([
                ['qual1', 'implies', 'param1'],
                
                ['qual1', 'implies', 'param2'],

                ['qual2', 'implies', 'param2'],
            ])

        expected_literals = np.array([
                ['qual1', 'implies', '-1.0'],
                ['param1', 'quantified by', '-1.0'],

                ['qual1', 'implies', '-1.0'],
                ['param2', 'quantified by', '-1.0'],

                ['qual2', 'implies', '-1.0'],
                ['param2', 'quantified by', '-1.0'],
            ])

        # Assert reality meets expectations
        self.assertTrue(np.array_equal(expected_literals, literal))
        self.assertTrue(np.array_equal(expected_relations, relations))