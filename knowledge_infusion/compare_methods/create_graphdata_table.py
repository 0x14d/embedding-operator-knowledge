"""
If compare methods was configured to create graph information (e.g closeness centrality, ...)
this module scans for the files puts the graph informations into Excel and Latex tables
"""

import pickle
import glob
import pandas as pd

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config import KnowledgeGraphGeneratorType


def create_graph_data_tables():
    result_path = "knowledge_infusion/compare_methods/results/debug/*/*/*/"
    exps = glob.glob(result_path)
    for exp_path in exps:
        paths = glob.glob(exp_path + "graph_data/*.pkl")
        train_files = []
        for path in paths:
            with open(path, "rb") as in_f:
                train_files.append(pickle.load(in_f))

        kgs = [kg.latex_label for kg in KnowledgeGraphGeneratorType]
        vals = list(train_files[0].keys())
        vals.remove('representation')
        df = pd.DataFrame(None, index=kgs, columns=vals)

        for file in train_files:
            for val in vals:
                kg_label = KnowledgeGraphGeneratorType(
                    file['representation']).latex_label
                df.at[kg_label, val] = file[val]

        with open(exp_path + "graph_data/graphdata.xlsx", "wb") as out_f:
            df.to_excel(out_f)
        with open(exp_path + "graph_data/graphdata.tex", "w") as out_f:
            df.to_latex(out_f, escape=False)


if __name__ == "__main__":
    create_graph_data_tables()
