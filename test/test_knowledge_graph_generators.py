import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd

from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.unqantified_knowledge_graph_generator import UnquantifiedKnowledgeGraphGenerator
from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.basic_knowledge_graph_generator import BasicKnowledgeGraphGenerator
from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.quantifed_parameters_without_shortcut import QuantifiedParametersWithoutShortcut
from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.quantified_parameters_with_shortcut import QuantifiedParametersWithShortcut
from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.quantified_parameters_with_literal import QuantifiedParametersWithLiteral
from data_provider.synthetic_data_generation.types.generator_arguments import KnowledgeGraphGeneratorArguments
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.config.modules.knowledge_graph_generator_config import EdgeWeightCalculationMethod


class TestKnowledgeGraphGenerators(unittest.TestCase):


    def setUp(self):
        """
        This sets up the tests for all knowledge graph generators.
        All input data is mocked away and a knowledge graph for three simple pqtuples are generated
        """

        # Mock the config
        config = Mock()
        config.knowledge_graph_generator = Mock()
        config.knowledge_graph_generator.edge_weight = EdgeWeightCalculationMethod.MEAN_ABSOLUTE
        config.knowledge_graph_generator.knowledge_share = 1
        config.knowledge_graph_generator.seed = 1

        quality_config = Mock()
        quality_config.max_rating = 2
        quality_config.min_rating = 1

        config.get_quality_by_name = MagicMock(return_value = quality_config)

        # Mock the tuples
        pq_tuples = Mock()
        pq_tuples.expert_knowledge = [('param1', 'qual1'), ('param2', 'qual1'), ('param2', 'qual2')]

        # Mock the functions
        def side_effect_function(value, last_parameter=None):
            return value
        function_obj = Mock()
        function_obj.inverse = MagicMock(side_effect = side_effect_function)
        inner_f = MagicMock()
        inner_f.get = MagicMock(return_value = function_obj)
        inner_f.__getitem__.side_effect = inner_f.get
        pq_functions = MagicMock()
        pq_functions.pq_functions = inner_f

        
        self.kggargs = KnowledgeGraphGeneratorArguments(
            sdg_config=config,
            pq_functions=pq_functions,
            pq_tuples=pq_tuples,
        )
    

    @staticmethod
    def generate_relations_list(generator):
        """
        Generates a list of relations present in a graph.
        This list can then be compared to the expected relations.
        """
        kg = generator.generate_knowledge_graph()

        edge_df = kg.get_edge_dataframe()
        vertex_df = kg.get_vertex_dataframe()

        print(edge_df)
        print(edge_df.to_numpy())
        print('------------------------')
        print(vertex_df)
        print(vertex_df.to_numpy())

        relations = []
        # Generate the relations list and replace ids with names
        for edge in edge_df.iterrows():
            id_target = int(edge[1]['target'])
            id_source = int(edge[1]['source'])
            rel = edge[1]['weight']

            target_str = vertex_df.iloc[id_target]['name']
            source_str = vertex_df.iloc[id_source]['name']
            relations.append([str(source_str), str(rel), str(target_str)])

        return relations


    def test_unquantified(self):
        """
        This function tests to unquantified representation
        """

        generator = UnquantifiedKnowledgeGraphGenerator(self.kggargs)
        relations = TestKnowledgeGraphGenerators.generate_relations_list(generator)
        # Expected Relations are always quality implies parameter in respect to the given pq tuples
        expected_relations = [
            ['qual1', 'implies', 'param1'],
            ['qual1', 'implies', 'param2'],
            ['qual2', 'implies', 'param2']
        ]
        for relation in expected_relations:
            self.assertTrue(relation in relations)

        self.assertTrue(len(expected_relations) == len(relations))


    def test_basic(self):
        """
        This function tests the basic representation
        """

        relations = TestKnowledgeGraphGenerators.generate_relations_list(BasicKnowledgeGraphGenerator(self.kggargs))

        # Expected relations are quality -value-> parameter
        # Value is always -1 due to the given min max ratings and the mocked function
        expected_relations = [
            ['qual1', '-1.0', 'param1'],
            ['qual1', '-1.0', 'param2'],
            ['qual2', '-1.0', 'param2']
        ]
        for relation in expected_relations:
            self.assertTrue(relation in relations)

        self.assertTrue(len(expected_relations) == len(relations))
    

    def test_quantified_with_shortcut(self):
        """
        This function tests the quantified_parameters_with_shortcut representation
        """

        relations = TestKnowledgeGraphGenerators.generate_relations_list(QuantifiedParametersWithShortcut(self.kggargs))

        # The simple relations from basic are splitted into three
        # Quality implies Parameter
        # Quality implies Value
        # Value quantifies Parameter
        expected_relations = [
            ['qual1', 'implies', 'param1'],
            ['qual1', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param1'],

            ['qual1', 'implies', 'param2'],
            ['qual1', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param2'],

            ['qual2', 'implies', 'param2'],
            ['qual2', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param2'],
        ]

        for relation in expected_relations:
            self.assertTrue(relation in relations)

        self.assertTrue(len(expected_relations) == len(relations))


    def test_quantified_without_shortcut(self):
        """
        This function tests the quantified_without_shortcut representation
        """

        relations = TestKnowledgeGraphGenerators.generate_relations_list(QuantifiedParametersWithoutShortcut(self.kggargs))

        # Same relation as in with_shortcut but the direct link between parameter and shortcut is missing
        expected_relations = [
            ['qual1', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param1'],

            ['qual1', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param2'],

            ['qual2', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param2'],
        ]

        for relation in expected_relations:
            self.assertTrue(relation in relations)

        self.assertTrue(len(expected_relations) == len(relations))


    def test_quantifed_with_literal(self):
        """
        This function tests the representation containing literals.
        """

        kgg = QuantifiedParametersWithLiteral(self.kggargs)
        graph = kgg.generate_knowledge_graph()

        edge_df = graph.get_edge_dataframe()
        vertex_df = graph.get_vertex_dataframe()

        # Since the edges are splitted into two seperate resulting list the conversion code already used had to be
        # expanded
        relations = []
        literals = []

        for edge in edge_df.iterrows():
            id_target = int(edge[1]['target'])
            id_source = int(edge[1]['source'])
            rel = edge[1]['weight']

            target_str = vertex_df.iloc[id_target]['name']
            source_str = vertex_df.iloc[id_source]['name']
            if edge[1]['literal_included'] == 'None':
                relations.append([str(source_str), str(rel), str(target_str)])
            else:
                literals.append([str(source_str), str(rel), str(target_str)])

    	# Same relation as in with shortcut, but relations leading to literal values should be able to get splitted of
        # based on the edge_identifying field 'literal_included'.
        expected_relations = [
            ['qual1', 'implies', 'param1'],

            ['qual1', 'implies', 'param2'],

            ['qual2', 'implies', 'param2'],
        ]
        expected_literals = [
            ['qual1', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param1'],

            ['qual1', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param2'],

            ['qual2', 'implies', '-1.0'],
            ['-1.0', 'quantifies', 'param2'],
        ]

        for relation in expected_relations:
            self.assertTrue(relation in relations)
        
        for literal in expected_literals:
            self.assertTrue(literal in literals)
        
        self.assertTrue(len(expected_relations) == len(relations))
        self.assertTrue(len(expected_literals) == len(literals))
        