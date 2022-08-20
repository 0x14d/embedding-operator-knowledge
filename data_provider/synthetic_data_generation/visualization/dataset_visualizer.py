"""
...
"""

# pylint: disable=relative-beyond-top-level, eval-used

from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Number
from typing import List, Optional, Union
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt

from ..config.basic_configs.parameter_config import ParameterProperties
from ..config.basic_configs.quality_config import QualityProperties
from ..types.experiments import GeneratedDataset

@dataclass
class DatasetVisualizer:
    """
    ...
    """
    dataset: GeneratedDataset

    output_directory: Optional[str] = None

    def __post_init__(self):
        if self.output_directory is not None:
            required_dirs = [
                self.output_directory,
                self.output_directory + 'quality_progress/',
                self.output_directory + 'pq_functions/'
            ]
            for directory in required_dirs:
                if not os.path.isdir(directory):
                    os.makedirs(directory)

    def plot_everything(self):
        """
        ...
        """
        self.plot_pq_functions()
        self.plot_parameters_per_quality()
        self.plot_quality_progress()

    def plot_pq_functions(self):
        """
        ...
        """
        qualities = self.dataset.pq_tuples.selected_qualities

        for quality in qualities:
            plot = self._create_plot()
            parameters = self.dataset.pq_tuples.get_parameters_affecting_quality(quality)
            for parameter in parameters:
                pq_function = self.dataset.pq_functions.pq_functions[(parameter, quality)]
                parameter_config = self.dataset.sdg_config.get_parameter_by_name(parameter)
                p_values = np.linspace(
                    start=parameter_config.min_value,
                    stop=parameter_config.max_value,
                    num=int((parameter_config.max_value - parameter_config.min_value) * 10)
                )
                q_values = pq_function(p_values)
                plot.add_plot(p_values, q_values)

            plot.add_quality_axis(quality) \
                .add_parameter_axis(parameters) \
                .add_title(quality) \
                .show() \
                .save(self.output_directory, f'pq_functions/{quality}.pdf')

    def plot_quality_progress(self):
        """
        ...
        """
        qualities = self.dataset.pq_tuples.selected_qualities

        for quality in qualities:
            plot = self._create_plot()
            for experiment_series in self.dataset.get_all_experiment_series_for_quality(quality):
                if len(experiment_series) == 0:
                    continue
                q_values = [e.qualities[quality] for e in experiment_series]
                x_values = range(len(q_values))
                plot.add_plot(x_values, q_values)
            plot.add_quality_axis(quality) \
                .add_axis('x', label='Iteration') \
                .add_title(quality) \
                .show() \
                .save(self.output_directory, f'quality_progress/{quality}.pdf')

    def plot_parameters_per_quality(self):
        """
        ...
        """
        qualities = self.dataset.pq_tuples.selected_qualities
        parameters = [len(self.dataset.pq_tuples.get_parameters_affecting_quality(q)) for q in qualities]
        self._create_plot() \
            .add_bar(qualities, parameters) \
            .show() \
            .save(self.output_directory, 'parameters_per_quality.pdf')

    def _create_plot(self) -> PlotCreator:
        return PlotCreator(
            dataset=self.dataset
        )

@dataclass
class PlotCreator:
    """
    ...
    """

    dataset: GeneratedDataset

    _axes: Optional[Axes] = None

    _is_smart_x_axis_initialized: bool = False
    _is_smart_y_axis_initialized: bool = False

    def __post_init__(self):
        _, self._axes = plt.subplots()

    def add_quality_axis(self, quality: Optional[str] = None) -> PlotCreator:
        """
        ...
        """
        self._axes.set_ylabel('Quality')

        if quality:
            quality_config = self.dataset.sdg_config.get_quality_by_name(quality)
        else:
            quality_config = QualityProperties()

        self._axes.set_ylim([quality_config.min_rating, quality_config.max_rating])

        return self

    def add_parameter_axis(self, parameter: Optional[Union[str, List[str]]] = None) -> PlotCreator:
        """
        ...
        """
        self._axes.set_xlabel('Parameter')

        if parameter is None:
            parameter_config = ParameterProperties()
        elif isinstance(parameter, str):
            parameter_config = self.dataset.sdg_config.get_parameter_by_name(parameter)
        else:
            parameter_config = ParameterProperties(
                min_value=min(
                        self.dataset.sdg_config.get_parameter_by_name(p).min_value for p in parameter
                    ),
                max_value=max(
                        self.dataset.sdg_config.get_parameter_by_name(p).max_value for p in parameter
                    )
            )

        self._axes.set_xlim([parameter_config.min_value, parameter_config.max_value])

        return self

    def add_axis(self, axis: str, label: Optional[str] = None) -> PlotCreator:
        """
        ...
        """
        if label is not None:
            getattr(self._axes, f'set_{axis}label')(label)

        return self

    def add_title(self, title: str) -> PlotCreator:
        """
        ...
        """
        self._axes.set_title(title)
        return self

    def add_plot(
        self,
        x_values,
        y_values,
        smart_x_axis: bool = True,
        smart_y_axis: bool = True
    ) -> PlotCreator:
        """
        ...
        """
        self._axes.plot(x_values, y_values)

        if smart_x_axis:
            x_min = min(x_values)
            x_max = max(x_values)
            self._smart_axis('x', x_min, x_max)

        if smart_y_axis:
            y_min = min(y_values)
            y_max = max(y_values)
            self._smart_axis('y', y_min, y_max)

        return self

    def add_bar(self, x_values, y_values) -> PlotCreator:
        """
        ...
        """
        self._axes.bar(x_values, y_values)

        if isinstance(x_values, str) or (not isinstance(x_values, Number) and isinstance(x_values[0], str)):
            plt.setp(self._axes.get_xticklabels(), rotation=30, horizontalalignment='right')

        
        return self

    def show(self) -> PlotCreator:
        """
        ...
        """
        plt.show()
        return self

    def save(self, output_directory: Optional[str], file_name: str) -> PlotCreator:
        if output_directory is not None:
            plt.savefig(output_directory + file_name)
        return self

    def _smart_axis(self, axis: str, min_value: Number, max_value: Number):
        if getattr(self, f'_is_smart_{axis}_axis_initialized'):
            lim = getattr(self._axes, f'get_{axis}lim')()
            lim = [min(lim[0], min_value), max(lim[1], max_value)]
        else:
            lim = [min_value, max_value]
            setattr(self, f'_is_smart_{axis}_axis_initialized', True)
        getattr(self._axes, f'set_{axis}lim')(lim)
