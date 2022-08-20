"""
This module provides the class `GaussianNoiseGenerator`.
"""

# pylint: disable=import-error, relative-beyond-top-level, too-few-public-methods

from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.modules.noise_generator_config \
    import GaussianNoiseGeneratorConfig
from data_provider.synthetic_data_generation.types.experiments \
    import GeneratedDataset, GeneratedExperiment
from data_provider.synthetic_data_generation.types.generator_arguments \
    import NoiseGeneratorArguments
from .abstract_noise_generator import NoiseGenerator

class GaussianNoiseGenerator(NoiseGenerator):
    """
    Class that provides functionality to add gaussian noise to a generated dataset.
    """

    _config: GaussianNoiseGeneratorConfig
    _rng: Generator

    def __init__(self, args: NoiseGeneratorArguments) -> None:
        super().__init__()
        self._config = args.sdg_config.noise_generator
        self._rng = rng(self._config.seed)

    def generate_noise(self, dataset: GeneratedDataset) -> None:
        """
        Generates noise on a dataset.
        """
        all_experiments = dataset.get_all_experiments()
        num_noised_experiments = int(self._config.noise_proportion * len(all_experiments))
        noised_experiments = self._rng.choice(
            a=all_experiments,
            size=num_noised_experiments,
            replace=False
        )

        for experiment in noised_experiments:
            experiment: GeneratedExperiment
            experiment.parameters = {
                k: v + self._gaussian_noise() for k,v in experiment.parameters.items()
            }
            experiment.qualities = {
                k: v + self._gaussian_noise() for k,v in experiment.qualities.items()
            }

    def _gaussian_noise(self) -> float:
        return self._rng.normal(loc=self._config.mean, scale=self._config.standard_deviation)
