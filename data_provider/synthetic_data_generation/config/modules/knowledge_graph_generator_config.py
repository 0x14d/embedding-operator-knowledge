"""
This module provides everything that is needed for the configuration of the
knowledge-graph-generators.

To add a new generator follow these steps:
1. Create a new class that inherits from `KnowledgeGraphGenerator` and implements the required
methods
(in a new file in the `data_provider/synthetic_data_generation/modules/knowledge_graph_generators`
directory)
2. Add a type name for the new generator to the `KnowledgeGraphGeneratorType` enum
3. Create a new config class for the new generator (in this module). It should inherit from
`AbstractKnowledgeGraphGeneratorConfig` and have an attribute `type` of datatype
`Literal[KnowledgeGraphGeneratorType.<New enum member (see step 2)>]`.
4. Add a new if case to the `get_configuration` method of `KnowledgeGraphGeneratorType` that returns
the new generator configuration if `self == KnowledgeGraphGeneratorType.<new enum member>`
5. Add the new config class to the `KnowledgeGraphGeneratorConfig` type by adding the new class to
the union (`Union[KnowledgeGraphGeneratorType, ..., <new config class>]`)
"""

# pylint: disable=no-name-in-module, import-error, import-outside-toplevel, too-few-public-methods

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field
from typing_extensions import Annotated

class KnowledgeGraphGeneratorType(str, Enum):
    """
    Enum that contains the key of every available knowledge-graph-generator.
    It's used in the configuration to specify what generator is used.

    When adding a new generator a new enum member must be added!
    """
    UNQUANTIFIED = 'unquantified'
    BASIC = 'basic'
    QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT = 'quantified_parameters_without_shortcut'
    QUANTIFIED_PARAMETERS_WITH_SHORTCUT = 'quantified_parameters_with_shortcut'
    QUANTIFIED_PARAMETERS_WITH_LITERAL = 'quantified_parameters_with_literal'
    QUANTIFIED_CONDITIONS = 'quantified_conditions'
    """Type of the `BasicKnowledgeGraphGenerator`"""

    def get_configuration(self) -> AbstractKnowledgeGraphGeneratorConfig:
        """
        Creates the matching configuration of the knowledge-graph-generator-type.

        Returns:
            knowledge-graph-generator configuaration with its default values
        """
        if self == KnowledgeGraphGeneratorType.BASIC:
            return BasicKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS:
            return QuantifiedConditionsKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.UNQUANTIFIED:
            return UnquantifiedKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_LITERAL:
            return QuantifiedParametersWithLiteralKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT:
            return QuantifiedParametersWithShortcutKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT:
            return QuantifiedParametersWithoutShortcutKnowledgeGraphGeneratorConfig()
        raise NotImplementedError(f'Missing implementation for type {self}!')

class EdgeWeightCalculationMethod(str, Enum):
    """
    Method to calculate the edge weights
    """
    MEAN_ABSOLUTE = 'mean_absolute'
    """Mean of all absolute changes of the parameter"""

class AbstractKnowledgeGraphGeneratorConfig(BaseModel, ABC):
    """
    Abstract class for the knowledge-graph-generator configurations.

    When adding a new generator the associated configuration class must inherit from this class!
    """

    @abstractmethod
    def get_generator_class(self) -> Any:
        """
        Returns the class of the knowledge-graph-generator this config is about.

        IMPORTANT: Import the returned class in this method and not outside of it!
        """


class BasicKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `BasicPQTupleGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.BASIC] = KnowledgeGraphGeneratorType.BASIC

    edge_weight: EdgeWeightCalculationMethod = EdgeWeightCalculationMethod.MEAN_ABSOLUTE
    """Method used to calculate the edge weights"""

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.knowledge_graph_generators. \
            basic_knowledge_graph_generator import BasicKnowledgeGraphGenerator
        return BasicKnowledgeGraphGenerator

class QuantifiedParametersWithoutShortcutKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `BasicPQTupleGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT

    edge_weight: EdgeWeightCalculationMethod = EdgeWeightCalculationMethod.MEAN_ABSOLUTE
    """Method used to calculate the edge weights"""

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.quantifed_parameters_without_shortcut import QuantifiedParametersWithoutShortcut
        return QuantifiedParametersWithoutShortcut

class QuantifiedParametersWithShortcutKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `BasicPQTupleGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT

    edge_weight: EdgeWeightCalculationMethod = EdgeWeightCalculationMethod.MEAN_ABSOLUTE
    """Method used to calculate the edge weights"""

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.quantified_parameters_with_shortcut import QuantifiedParametersWithShortcut
        return QuantifiedParametersWithShortcut

class QuantifiedParametersWithLiteralKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `BasicPQTupleGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_LITERAL] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_LITERAL

    edge_weight: EdgeWeightCalculationMethod = EdgeWeightCalculationMethod.MEAN_ABSOLUTE
    """Method used to calculate the edge weights"""

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.quantified_parameters_with_literal import QuantifiedParametersWithLiteral
        return QuantifiedParametersWithLiteral

class UnquantifiedKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `BasicPQTupleGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.UNQUANTIFIED] = KnowledgeGraphGeneratorType.UNQUANTIFIED

    edge_weight: EdgeWeightCalculationMethod = EdgeWeightCalculationMethod.MEAN_ABSOLUTE
    """Method used to calculate the edge weights"""

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.knowledge_graph_generators.unqantified_knowledge_graph_generator import UnquantifiedKnowledgeGraphGenerator
        return UnquantifiedKnowledgeGraphGenerator


class QuantifiedConditionsKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedQualitiesTupleGenerarator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 3

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.knowledge_graph_generators. \
            quantified_conditions_knowledge_graph_generator import QuantifiedConditionsKnowledgeGraphGenerator
        return QuantifiedConditionsKnowledgeGraphGenerator


KnowledgeGraphGeneratorConfig = Annotated[
                                    Union[
                                        KnowledgeGraphGeneratorType,
                                        BasicKnowledgeGraphGeneratorConfig,
                                        QuantifiedConditionsKnowledgeGraphGeneratorConfig,
                                        UnquantifiedKnowledgeGraphGeneratorConfig,
                                        QuantifiedParametersWithoutShortcutKnowledgeGraphGeneratorConfig,
                                        QuantifiedParametersWithShortcutKnowledgeGraphGeneratorConfig,
                                        QuantifiedParametersWithLiteralKnowledgeGraphGeneratorConfig,
                                    ],
                                    Field(discriminator='type')
                                ]
"""Type alias that contains all available knowledge-graph-generator configuration classes."""


DEFAULT_KNOWLEDGE_GRAPH_GENERATOR_CONFIG: KnowledgeGraphGeneratorType = \
    KnowledgeGraphGeneratorType.BASIC
"""Default knowledge-graph-generator type that is used if no configuration is provided."""


def parse_knowledge_graph_generator_config(
    config: Optional[KnowledgeGraphGeneratorConfig]
) -> AbstractKnowledgeGraphGeneratorConfig:
    """
    Parses a knowledge-graph-generator configuration to its default format.

    If the configuration is None the default configuration is used.

    Parameters:
        config (KnowledgeGraphGeneratorConfig | None): config that should be parsed

    Returns:
        config in the default format (AbstractKnowledgeGraphGeneratorConfig)
    """
    if config is None:
        config = DEFAULT_KNOWLEDGE_GRAPH_GENERATOR_CONFIG
    if isinstance(config, KnowledgeGraphGeneratorType):
        return config.get_configuration()
    return config
