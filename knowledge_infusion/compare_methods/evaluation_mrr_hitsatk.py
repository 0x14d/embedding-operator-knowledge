
import pickle
import glob
import pandas as pd
import os
from statistics import mean, stdev

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config import KnowledgeGraphGeneratorType
from knowledge_infusion.compare_methods.configs.compare_methods_config import AmriConfig
from knowledge_infusion.graph_embeddings.embedding_types import EmbeddingType

def load_data(path: str):
    with open(path + 'results.pickle', 'rb') as inputf:
        data = pickle.load(inputf)
    
    return data

def create_df(data, side):
    rows = [EmbeddingType(e).latex_label for e in AmriConfig().embedding_types]
    kgs = [kg.latex_label for kg in KnowledgeGraphGeneratorType]
    columns = pd.MultiIndex.from_product([kgs, ['AMRI', 'Hits@1', 'Hits@5', 'Hits@10']], names=['Representations', 'Metrics'])
    df = pd.DataFrame(None, index=rows, columns=columns)

    for measurement in data:
        # Sometimes the values in the data are not Numbers (e.g. "N/A", "1.34±0.01") in those cases round() will fail
        # and instead we directly write the value into our dataframe.
        method = measurement['method'].latex_label
        kg_type = measurement['kg_type'].latex_label
        try:
            df.at[method, (kg_type, 'AMRI')] = round(measurement['results']['adjustedarithmeticmeanrankindex', side], 2)
            df.at[method, (kg_type, 'Hits@1')] = round(measurement['results']['hits_at_1', side], 2)
            df.at[method, (kg_type, 'Hits@5')] = round(measurement['results']['hits_at_5', side], 2)
            df.at[method, (kg_type, 'Hits@10')] = round(measurement['results']['hits_at_10', side], 2)
        except TypeError:
            df.at[method, (kg_type, 'AMRI')] = measurement['results']['adjustedarithmeticmeanrankindex', side]
            df.at[method, (kg_type, 'Hits@1')] = measurement['results']['hits_at_1', side]
            df.at[method, (kg_type, 'Hits@5')] = measurement['results']['hits_at_5', side]
            df.at[method, (kg_type, 'Hits@10')] = measurement['results']['hits_at_10', side]


    return df.transpose()

def save_all_tables(df, data_folder, filename_specifier):
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    with open(data_folder + 'table'+ filename_specifier +'.csv', 'w') as f:
        df.to_csv(f, sep=";")
    with open(data_folder + 'table'+ filename_specifier +'.xlsx', 'wb') as f:
        df.to_excel(f)
    with open(data_folder + 'table'+ filename_specifier +'.tex', 'w') as f:
        df.to_latex(f)

def combine_multiple_iterations(iteration_results, with_std = False):
    og = iteration_results[0]

    for result in og:
        # Find matching results from different metrics
        matches = []
        for i in range(1, len(iteration_results)):
            current = iteration_results[i]
            for element in current:
                if result['kg_type'] == element['kg_type'] and result['method'] == element['method']:
                    matches.append(element)
        
        # Combine the results
        for key in result['results'].keys():
            similar = []
            for element in matches:
                similar.append(element['results'][key])
            mean_s = mean(similar)
            if with_std:
                result['results'][key] = str(round(mean_s, 2)) + '±' + str(round(stdev(similar), 2))
            else:
                result['results'][key] = round(mean_s, 2)
    
    return og
            

if __name__ == '__main__':
    result_path = "knowledge_infusion/compare_methods/results/*/"
    exps = glob.glob(result_path)

    for base_path in exps:
        number_iters = len(glob.glob(base_path + 'iteration*'))

        paths = []
        collected_data = []

        for i in range(number_iters):
            paths.append(base_path + "iteration" + str(i) + "/")
        for path in paths:
            save_path = path + "eval/"
            data = load_data(path)
            frame =create_df(data, 'head')
            save_all_tables(frame, save_path, 'head')

            frame = create_df(data, 'tail')
            save_all_tables(frame, save_path, 'tail')

            collected_data.append(data)
        
        multi = combine_multiple_iterations(collected_data)
        frame = create_df(multi, 'head')
        save_all_tables(frame, base_path + "combined_eval_results_no_std/", 'head')

        frame = create_df(multi, 'tail')
        save_all_tables(frame, base_path + "combined_eval_results_no_std/", 'tail')

        multi = combine_multiple_iterations(collected_data, with_std=True)
        frame = create_df(multi, 'head')
        save_all_tables(frame, base_path + "combined_eval_results_with_std/", 'head')

        frame = create_df(multi, 'tail')
        save_all_tables(frame, base_path + "combined_eval_results_with_std/", 'tail')
