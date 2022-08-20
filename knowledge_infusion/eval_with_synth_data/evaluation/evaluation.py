"""
...
"""

# pylint: disable=relative-beyond-top-level, import-error

from dataclasses import dataclass, field
import shutil
import re
import os
from typing import Any, Dict, Optional, Union
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..experiments.experiment_config import ExperimentSetup, ExperimentConfig
from data_provider.synthetic_data_generation.synthetic_data_generator2 import SyntheticDataGenerator
from data_provider.synthetic_data_generation.visualization.dataset_visualizer import DatasetVisualizer

@dataclass
class ExperimentEvaluation:
    """
    ...
    """

    experiment_config: ExperimentSetup = ExperimentSetup()
    """..."""

    output_directory: str = 'knowledge_infusion/eval_with_synth_data/evaluation/evaluation/'
    """..."""

    # Style
    method_labels: Dict[str, str] = field(default_factory=lambda: {
        "fnn": "FNN",
        "sss": "SSS",
        "ckl": "LCK"
    })
    """..."""

    boxplot_legend_label: str = 'Approach'
    """..."""

    boxplot_x_labels: Dict[int, str] = field(default_factory=lambda: {
        1: '',
        2: '',
        5: ''
    })

    def __post_init__(self):
        # Copy and rename result files
        def copy_result_file(src, dest_dir, experiment_key):
            if not os.path.isdir(dest_dir):
                os.makedirs(dest_dir)
            shutil.copy(
                src,
                f'{dest_dir}{experiment_key}.json'
            )

        def find_result_file(directory: str) -> Optional[str]:
            if not os.path.isdir(directory):
                return None
            result_file_regex = re.compile('^result.*json$')
            (_, _, filenames) = next(os.walk(directory))
            try:
                return [f for f in filenames if result_file_regex.match(f)][0]
            except IndexError:
                return None

        if os.path.isdir(self.output_directory):
            shutil.rmtree(self.output_directory)

        for method in self.experiment_config.methods:
            default_result_directory = \
                f'{self.experiment_config.output_directory}experimentdefault/{method}/'
            default_result_file = find_result_file(default_result_directory)
            for experiment, experiment_data in self.experiment_config.experiments.items():
                if experiment == 'default':
                    continue
                src_directory = f'{self.experiment_config.output_directory}experiment{experiment}/{method}/'
                if not os.path.isdir(src_directory):
                    continue
                for key in experiment_data.keys:
                    src_directory_with_key = f'{src_directory}{key}/'
                    result_file = find_result_file(src_directory_with_key)
                    copy_result_file(
                        src_directory_with_key + result_file,
                        self.output_directory + f'experiment{experiment}/results/{method}/',
                        key
                    )
                if default_result_file is not None and experiment_data.default_key is not None:
                    copy_result_file(
                        default_result_directory + default_result_file,
                        self.output_directory + \
                        f'experiment{experiment}/results/{method}/',
                        experiment_data.default_key
                    )

    def generate_boxplots(self):
        """
        ...
        """
        def generate_experiment_key_df(experiment_id: int, key: Any) -> pd.DataFrame:
            result_df = pd.DataFrame()
            for method in self.experiment_config.methods:
                directory = f'{self.output_directory}experiment{experiment_id}/results/{method}/'
                result_file = f'{directory}{key}.json'
                with open(result_file) as file:
                    results = json.load(file)
                df = pd.DataFrame([i['validation_result']['mse'] for i in results['folds'] if i != "time_needed"])
                df['arch'] = [self.method_labels[method] for _ in range(len(results['folds']))]
                result_df = pd.concat([result_df, df], axis=0)
            return result_df

        def generate_experiment_df(experiment_id: int, experiment_data: ExperimentConfig):
            result_df = pd.DataFrame()
            for key in experiment_data.all_keys:
                df = generate_experiment_key_df(experiment_id, key)
                df.set_axis([str(key), self.boxplot_legend_label], axis=1, inplace=True)
                df = df.melt(self.boxplot_legend_label, var_name='type', value_name='vals')
                result_df = pd.concat([result_df, df], axis=0)
            return result_df

        for experiment_id, experiment_data in self.experiment_config.experiments.items():
            if not os.path.isdir(f'{self.output_directory}experiment{experiment_id}'):
                continue
            df = generate_experiment_df(experiment_id, experiment_data)
            sns.boxplot(
                data=df, x='type', y='vals', hue=self.boxplot_legend_label, showmeans=True,
                meanprops={
                    "markerfacecolor":"#3D3D3D", "markeredgecolor":"#3D3D3D", "marker": "o", "markersize": 5
                }
            )
            plt.ylabel("MSE Loss")
            plt.xlabel(self.boxplot_x_labels.get(experiment_id, ''))
            plt.savefig(f'{self.output_directory}experiment{experiment_id}/boxplot.pdf')
            plt.clf()

    def generate_visualizations(self):
        """
        ...
        """
        def visualization(config: str, experiment_id: Union[str, int], key: Any):
            ds = SyntheticDataGenerator(config).create_dataset()
            visualizer = DatasetVisualizer(
                dataset=ds,
                output_directory=f'{self.output_directory}experiment{experiment_id}/visualization/{key}/'
            )
            visualizer.plot_everything()

        for experiment_id, experiment_data in self.experiment_config.experiments.items():
            if not os.path.isdir(f'{self.output_directory}experiment{experiment_id}'):
                continue
            for key in experiment_data.keys:
                config = f'{self.experiment_config.sdg_config_directory}experiment{experiment_id}/{key}-config.json'
                visualization(config, experiment_id, key)
            if experiment_data.default_key is not None:
                visualization(
                    self.experiment_config.default_sdg_config_path,
                    experiment_id,
                    experiment_data.default_key
                )
            
            

def main():
    evaluation = ExperimentEvaluation()
    evaluation.generate_boxplots()
    evaluation.generate_visualizations()

if __name__ == '__main__':
    main()
