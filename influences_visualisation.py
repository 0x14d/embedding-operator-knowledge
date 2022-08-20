import os
from collections import defaultdict

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import experiment_definition as ed
import knowledge_graph as kg
from utilities import camel_to_snake_ratings


def create_directories_if_necessary(filename):
    """
    Creates directories in the given filepath
    :param filename: path containing directories that maybe have to be created
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


class InfluenceVisualiser:
    @staticmethod
    def plot_graph(params_dict: dict, title: str = None, show_weights: bool = False):
        """
        Plots a knowledge graph of what influences influenced what parameters
        :param params_dict dictionary with parameters for this function --> requires a key 'filtered_graph'
        and a key 'file_name'
        :param title title for the plot
        :param show_weights
        """
        # check if the necessary keys are in the dict, otherwise leave function without plotting anything
        if not ('filtered_graph' in params_dict.keys() and 'file_name' in params_dict.keys()):
            print('Necessary parameters not given --> no graph will be plotted')
            return
        graph = params_dict['filtered_graph']
        file_name = params_dict['file_name']
        color_dict = {"qual_influence": "blue", "env_influence": "brown", "parameter": "yellow"}
        visual_style = {}
        visual_style["vertex_size"] = 20
        # attributes are not a propper dictionary that means we have to revert to nonstandard methods to ignore nonexisting properties than occur i.e. if graph has no vertices - in that case we abort plotting
        try:
            visual_style["vertex_color"] = [color_dict[type] for type in graph.vs["type"]]
            visual_style["vertex_label"] = graph.vs["name"]
            if show_weights:
                visual_style["edge_label"] = [round(w, 2) for w in graph.es["weight"]]
            min_weight = min(graph.es['weight'])
            max_weight = max(graph.es['weight'])
            # edge_weights_normalized = [(w - min_weight) / (max_weight - min_weight) * 5 for w in graph.es['weight']]

            visual_style["edge_width"] = [1 + weight for weight in graph.es['weight']]
            visual_style["edge_color"] = "rgba(1,1,1,0.1)"
        except KeyError as e:
            print("Encountered " + str(e) + "while plotting")
        visual_style["vertex_label_dist"] = 1
        visual_style["layout"] = graph.layout_kamada_kawai()
        if "filtered" in file_name:
            visual_style["bbox"] = (1500, 1500)
            visual_style["margin"] = 100
        else:
            visual_style["bbox"] = (1700, 1700)
            visual_style["margin"] = 60
        create_directories_if_necessary(file_name)
        plt = ig.plot(graph, file_name, **visual_style)
        if title:
            plt.title = title

    @staticmethod
    def plot_envfactor_parameter_pair(params_dict: dict, quality: str = "quality_aggregated"):
        """
        :param params_dict --> dict with keys 'filtered_graph' and 'exp_series' necessary
        :param quality
        """
        quality_names = {"quality_aggregated": "Aggregated Quality", "stringing": "Quality Strining"}
        param_names = {"retraction_count_max": "Maximum Retraction Count",
                       "retraction_extrusion_window": "Retraction Extrusion Window"}
        if not ('filtered_graph' in params_dict.keys() and 'exp_series' in params_dict.keys()):
            print('Necessary parameters not given --> no graph will be plotted')
            return
        es_ps = kg.extract_e_p_values(params_dict['filtered_graph'], params_dict['exp_series'], fill_nones=False)
        for e_p in es_ps:
            parameter, influence = e_p.split('-')
            df = es_ps[e_p]
            if not df.empty:
                # g = sns.FacetGrid(df, col="influence_value", hue="series_id", col_wrap=2, legend_out=True)
                # g.map_dataframe(sns.stripplot, x="param_value", y=quality, jitter=False)
                g = sns.catplot(x="param_value", y=quality, hue="series_id",
                                col="influence_value", data=df, col_wrap=2, legend=False)
                g.set_titles(col_template=influence + " = {col_name}")
                g.set_axis_labels(parameter, quality_names[quality])
                # cols = df.columns
                # df[cols] = df[cols].apply(pandas.to_numeric, errors='ignore')
                # ax = sns.lmplot(x="param_value", y="quality_aggregated", data=df, fit_reg=False, hue="series_id",col="influence_value",col_wrap=2, height=3)
                # # ax = sns.scatterplot(data=df, x="param_value", y="quality_aggregated", hue="series_id", style='influence_value')
                # ax.set(xlabel=parameter, ylabel="quality_aggregated")
                # ax.set_title(parameter + ' - ' + influence)
                g.add_legend(title="Task", bbox_to_anchor=(0.98, 0.3), borderaxespad=0.)
                # plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
                # g.fig.get_axes()[0].legend(loc='lower left')
                plt.tight_layout()
                filename = 'figs/eps/' + influence + '-' + parameter + '.pdf'
                create_directories_if_necessary(filename)
                plt.savefig(filename, dpi=600)

    @staticmethod
    def plot_all_envfactor_parameter_pairs_relative(knowledge_graph, experiment_series, influence='', envfactor=''):
        """
        generates all plots for influences and the process parameters that have changed and how they improved the error

        call this function like plot_all_envfactor_parameter_pair_relative_influence(knowledge_graph, experiment_series, influence='warping')
        to get all plots for warping shown

        call this function like plot_all_envfactor_parameter_pair_relative_influence(knowledge_graph, experiment_series) if you like all plots for all influences
        to be new generated

        :param knowledge_graph: graph that contains all the vertices and edges
        :param experiment_series: all experiment series that contain the data you want to plot
        :param influence: if this parameter is set to any influence the given graph will be filtered and only this influence will be considered
        """

        influences = set()  # all influences we find in the data
        graph = InfluenceVisualiser._try_filter_knowledge_graph(knowledge_graph, influence)

        params_dict = dict()
        params_dict['filtered_graph'] = graph
        params_dict['exp_series'] = experiment_series
        params_dict['relative_parameter'] = False
        influences = InfluenceVisualiser.plot_envfactor_parameter_pair_relative_influence(params_dict)
        params_dict['relative_parameter'] = True
        InfluenceVisualiser.plot_envfactor_parameter_pair_relative_influence(params_dict)

        for influence_to_filter in influences:
           filtered_kg = kg.filter_knowledge_graph_for_influence(knowledge_graph, influence_to_filter)
           filename = f'figs/{camel_to_snake_ratings(influence_to_filter)}/filtered_v' + str(
               len(filtered_kg.vs)) + '_e' + str(
               len(filtered_kg.es)) + '.pdf'
           create_directories_if_necessary(filename)
           try:
               InfluenceVisualiser.plot_graph(filtered_kg, fname=filename)
           except:
               print(f'something went wrong for plotting graph {influence_to_filter}')

    @staticmethod
    def _plot_ep_detail(parameter, parameters_df, influence, influence_dfs):
        plt.clf()
        parameters_df[parameters_df.columns] = parameters_df[parameters_df.columns].apply(pandas.to_numeric, errors='ignore')
        influence_dfs[influence_dfs.columns] = influence_dfs[influence_dfs.columns].apply(pandas.to_numeric, errors='ignore')
        ax = sns.regplot(x=parameters_df[parameter], y=influence_dfs[influence], fit_reg=False, y_jitter=0.2)
        ax.set(xlabel="$\\Delta$ " + parameter, ylabel=f"$\\Delta$ {influence}")
        filename = f'figs/{influence}/eps/rel/' + influence + '-' + parameter + '.pdf'
        print(f'mean:{np.mean(parameters_df[parameter])}')
        plt.tight_layout()
        create_directories_if_necessary(filename)
        plt.savefig(filename, dpi=600)

        plt.show()
        plt.clf()

    @staticmethod
    def plot_envfactor_parameter_pair_relative_to_preceeding_exp(parameters, influences):
        """
        Computes Visualisations for relative quality improvements obtained by relative parameter adjustments
        :param parameters:
        :param influences:
        :return:
        """
        for influence in influences.columns:
            for parameter in parameters.columns:
                InfluenceVisualiser._plot_ep_detail(parameter, parameters, influence, influences)

    @staticmethod
    def plot_envfactor_parameter_pair_relative_influence(params_dict: dict, legend=False) -> set:
        """
        based on df of e-p pairs computes visualisations for (relative) quality improvements produced by relative parameter adjustments
        Watch out: Plots can not be generated for e_p pairs that only have one experiment id in their series
        :param params_dict
        :param legend:
        :return:
        """
        influences = set()
        if not ('filtered_graph' in params_dict.keys() and 'exp_series' in params_dict.keys()
                and 'relative_parameter' in params_dict.keys()):
            print('Necessary parameters not given --> no graph will be plotted')
            return influences
        relative_parameter = params_dict['relative_parameter']
        if relative_parameter is True:
            fill_nones = True
        else:
            fill_nones = False
        es_ps = kg.extract_e_p_values(params_dict['filtered_graph'], params_dict['exp_series'], fill_nones=fill_nones)
        for e_p in es_ps:
            parameter, influence = e_p.split('-')
            influence = camel_to_snake_ratings(influence)
            influences.add(influence)
            df = es_ps[e_p]
            plt.clf()
            if not df.empty:
                # multiply with -1 so that a positive value means a higher error in the influence
                # e.g. stl file has stringing 1 and after changing one parameter it has stringing 2
                # in the diagram it shall be demonstrated with a +1 higher point
                try:
                    # as we don't have data for all possible influences we have to catch exceptions here
                    df[f'relative_{influence}'] = -1 * (df[influence] - df['influence_value'])
                except:
                    continue
                df = df.rename(columns={"series_id": "Task Id"}, errors="raise")
                cols = df.columns
                df[cols] = df[cols].apply(pandas.to_numeric, errors='ignore')
                if relative_parameter is True:
                    try:
                        df['relative_param'] = df.param_value - df.param_value_original
                        df.select_dtypes(['number'])
                    except Exception:
                        # TODO XAI-587 right way to deal with params that have values like 'True' or 'False'?
                        continue
                    # ax = sns.scatterplot(data=df, x="relative_param", y=f"relative_{influence}", hue="Task Id",
                    #                      #                                 style="Task Id",
                    #                      alpha=0.6, x_jitter=0.2, y_jitter=0.2, edgecolor=None)
                    try:
                        ax = sns.regplot(data=df, x="relative_param", y=f"relative_{influence}", fit_reg=False, y_jitter = 0.2)
                    except:
                        print(f'Plot could not be generated for {influence}-{parameter}')
                        continue
                    ax.set(xlabel="$\\Delta$ " + parameter, ylabel=f"$\\Delta$ {influence}")
                    filename = f'figs/{influence}/eps/rel/' + influence + '-' + parameter + '.pdf'
                    print(f'mean:{np.mean(df.relative_param)}')
                else:
                    # ax = sns.scatterplot(data=df, x="param_value", y=f"relative_{influence}", hue='Task Id',
                    #                      alpha=0.6, x_jitter=0.2, y_jitter=0.2, edgecolor=None)
                    df.select_dtypes(['number'])
                    try:
                        ax = sns.regplot(data=df, x="param_value", y=f"relative_{influence}", fit_reg=False, y_jitter = 0.2)
                    except:
                        print(f'Plot could not be generated for {influence}-{parameter}')
                        continue
                    # g = sns.FacetGrid(df, col="influence_value", hue="series_id", col_wrap=2, legend_out=True)
                    ax.set(xlabel=parameter, ylabel=f"$\\Delta$ {influence}")
                    filename = f'figs/{influence}/eps/' + influence + '-' + parameter + '.pdf'
                    try:
                        print(f'mean:{np.mean(df.param_value)}')
                    except:
                        print('Categorical value: mean value can not be calculated')
                # ax.get_legend().remove()
                plt.tight_layout()
                create_directories_if_necessary(filename)
                plt.savefig(filename, dpi=600)

                plt.show()
                plt.clf()
        return influences

    @staticmethod
    def plot_preceeding_experiment(params_dict: dict):
        """
        Plot the change of qualities not regarding to the standard parametrisation but regarding to the previous
        experiment
        :param params_dict dictionary with the parameters --> necessary keys are 'filtered_graph' which is a weighted
        and filtered aggregated knowledge graph and 'exp_series' which contains the experiment series regarding to the
        knowledge graph
        """
        if not ('filtered_graph' in params_dict.keys() and 'exp_series' in params_dict.keys()):
            print('Did not receive necessary arguments --> will not plot anything')
            return
        graph = params_dict['filtered_graph']
        exp_series = params_dict['exp_series']

        relative_parameters = InfluenceVisualiser.dataframe_preparation(graph, exp_series,
                                                                        ed.WeightingClustering.CLUSTERING_PARAMETER_RELATIVE,
                                                                        active_fill=False)
        relative_qualities = InfluenceVisualiser.dataframe_preparation(graph, exp_series,
                                                                       ed.WeightingClustering.CLUSTERING_QUALITY_RELATIVE,
                                                                       active_fill=False)
        InfluenceVisualiser.plot_envfactor_parameter_pair_relative_to_preceeding_exp(relative_parameters, relative_qualities)

    @staticmethod
    def weighting_relations_level(knowledge_graph, experiment_series, df_computation_fct, active_fill=True,
                                  quality_relative=False):
        """
        :param knowledge_graph:
        :param experiment_series:
        :param quality_relative: whether to use quality improvement(relative) or absolute quality
        :return:
        """
        influence_df, quality_df = InfluenceVisualiser.dataframe_preparation(knowledge_graph, experiment_series,
                                                                             df_computation_fct, active_fill)

        # compute it for relative (=quality improvement) and absolute quality
        weighted_graph = knowledge_graph.copy()
        relations = weighted_graph.es
        for relation in relations:
            experiments = relation['experiments']
            weight_per_experiment = []
            for experiment in experiments:
                quality_aggregated = quality_df.loc[experiment, :].sum()

                quality_influences_keys = quality_df.columns
                # only consider quality influences
                for influence_key in influence_df.columns:
                    if influence_key not in quality_influences_keys:
                        del influence_df[influence_key]
                quality_influences_summed = influence_df.loc[experiment, :].dropna().sum()
                weight_per_experiment.append(quality_influences_summed/quality_aggregated)
            quality = sum(weight_per_experiment) / len(weight_per_experiment)
            if not np.isnan(quality) and not np.isinf(quality):
                relation['weight'] = quality
            else:
                # if we had a division by 0 we set the weight to 0 since both dividend & divisor are 0
                relation['weight'] = 0

        return weighted_graph

    @staticmethod
    def weighting_double_clustering(knowledge_graph, experiment_series, df_computation_fct, active_fill=False,
                                    min_samples_per_cluster=2):
        """
        Weights the graph according to an experiment being successfully clustered in two spaces e.g. parameter and quality.
        :param min_samples_per_cluster: the minimum amount of sampes that leads OPTICS to determine a cluster
        :param active_fill: whether to fill NaNs in the DFs by referring to the lookup dictionaries - this increases the amount of data at the cost of muddifying the assumption that only influences highlighted by users are relevant for a given experiment
        :param df_computation_fct: is expected to create a tuple of the two DFs that are then clustered
        :param knowledge_graph:
        :param experiment_series:
        :return:
        """
        parameter_df, quality_df = InfluenceVisualiser.dataframe_preparation(knowledge_graph, experiment_series,
                                                                             df_computation_fct, active_fill)

        parameter_df = parameter_df.fillna(-5)
        quality_df = quality_df.fillna(-5)

        parameter_df = pandas.DataFrame(StandardScaler().fit_transform(parameter_df), index=parameter_df.index, columns=parameter_df.columns)
        quality_df = pandas.DataFrame(StandardScaler().fit_transform(quality_df), index=quality_df.index, columns=quality_df.columns)

        clusters_param, expt_to_cluster_param, cluster_to_exp_param = clustering_visualisation(parameter_df, min_samples=min_samples_per_cluster)
        cluster_qual, exp_to_cluster_qual, cluster_to_exp_qual = clustering_visualisation(quality_df, min_samples=min_samples_per_cluster)

        exp_to_both_clusters = dict()
        # check if experiment is assigned a cluster in both, parameter and quality space
        for experiment, value in expt_to_cluster_param.items():
            if experiment in exp_to_cluster_qual:
                if exp_to_cluster_qual[experiment] >= 0:
                    exp_to_both_clusters[experiment] = 1
                else:
                    exp_to_both_clusters[experiment] = -1
            else:
                exp_to_both_clusters[experiment] = -1
        # do the check the other way rond in case one df contains more experiments than the other
        for experiment, value in exp_to_cluster_qual.items():
            if experiment in expt_to_cluster_param:
                if expt_to_cluster_param[experiment] >= 0:
                    exp_to_both_clusters[experiment] = 1
                else:
                    exp_to_both_clusters[experiment] = -1
            else:
                exp_to_both_clusters[experiment] = -1

        weighted_graph = kg.weight_aggregated_graph(knowledge_graph, exp_to_both_clusters)
        return weighted_graph

    @staticmethod
    def dataframe_preparation(knowledge_graph, experiment_series, df_computation_fct, active_fill):
        dfs = kg.prepare_dfs_of_experiments(knowledge_graph, experiment_series,
                                            fill_nones=active_fill)  # get user given insights and changed process parameters from the given graph
        filled_dfs = {
            key: kg.fill_nans_of_clustering_df(graph=knowledge_graph, df=df, type=key, active_fill=active_fill) for
            key, df in dfs.items()}
        df = df_computation_fct(filled_dfs)
        return df

    @staticmethod
    def weight_clustering(knowledge_graph, experiment_series, df_computation_fct, active_fill=False):
        """
        # TODO refactor - shouldn't be part of influence visualisation
        :param knowledge_graph:
        :param experiment_series:
        :return:
        """
        # TODO move aggregations to a seperate file
        dfs = kg.prepare_dfs_of_experiments(knowledge_graph, experiment_series,
                                            fill_nones=active_fill)  # get user given insights and changed process parameters from the given graph
        filled_dfs = {key: kg.fill_nans_of_clustering_df(graph=knowledge_graph, df=df, type=key, active_fill=active_fill) for key, df in dfs.items()}
        # TODO XAI-558 refactor! use _dataframe_preparation!
        InfluenceVisualiser.__enocde_categorical_values(filled_dfs)
        filled_dfs = {key: filled_df.fillna(-5) for key, filled_df in filled_dfs.items()}


        # append the key (e.g. 'influence') so we do not have column collissions/overwriting (occuring by including the same columns e.g. through infleucnes & quality) in pandas.concat
        for key, df in filled_dfs.items():
            renaming_dict = {name: key + '-' + name for name in df.columns}
            filled_dfs[key] = df.rename(renaming_dict, axis=1)

        df = df_computation_fct(filled_dfs)

        # norm
        normed_df = pandas.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)

        # from sknetwork.clustering import Louvain
        # louvain = Louvain()
        # cluster_louvain = louvain.fit_transform(adjacency)
        from scipy.spatial.distance import minkowski
        clusters = dict(zip(normed_df.index, OPTICS(min_samples=2, metric=minkowski).fit(normed_df).labels_))
        weighted_graph = kg.weight_aggregated_graph(knowledge_graph, clusters)
        return weighted_graph

    @staticmethod
    def filter_graph(weighted_graph, threshold_function):
        filtered_gs = kg.filter_aggregated_graph(weighted_graph, threshold_function)
        #remove nodes with no neighbors
        isolated_vertices = filtered_gs.vs.select(_degree=0)
        filtered_gs.delete_vertices(isolated_vertices)
        return filtered_gs

    # @deprecation.deprecated(details="should not be neccessary anymore since label encoding is applied earlier - check and delete")
    @staticmethod
    def __enocde_categorical_values(filled_dfs):
        """
        in place modification of all columns containing string values
        TODO refactor to not mutate filled_Dfs
        :param filled_dfs:
        :return:
        """
        for key, df in filled_dfs.items():
            for column in filled_dfs[key].columns:
                # if we have a categorical column we should label encode it
                if filled_dfs[key][column].dtype == np.str or filled_dfs[key][column].dtype == np.object:
                    # TODO cache label encodings so we don't have to retrain them each time (however right now we have 4 encodings so probably not worth the effort)
                    le = preprocessing.LabelEncoder()  # TODO switch to ordinal encoder to specify ordering
                    # mask it to be able restore nans
                    original = filled_dfs[key][column]
                    mask = filled_dfs[key][column].isnull()
                    # cast to string to get around encoding nan error
                    ids = le.fit_transform(filled_dfs[key][column].astype(str).values)
                    transformed_column = filled_dfs[key][column].astype(str).apply(lambda x: le.transform([x])[0])
                    transformed_column = transformed_column.where(~mask, original)  # restore nans
                    filled_dfs[key][column].update(transformed_column)
                    mapping = dict(zip(le.classes_, range(len(le.classes_))))
                    with open(f'label_encoding_lookups/labelencoding_df_{key}_column_{column}.csv', 'w') as f:
                        import csv
                        w = csv.DictWriter(f, mapping.keys())
                        w.writeheader()
                        w.writerow(mapping)


    @staticmethod
    def _try_filter_knowledge_graph(knowledge_graph, influence):
        if len(influence) > 0:
            # for debugging purposes: we only generate plots for one specific influence
            graph = kg.filter_knowledge_graph_for_influence(knowledge_graph, influence)
        else:
            graph = knowledge_graph  # we can use the whole graph to get influences
        return graph


def cluster_quality_visualisation(knowledge_graph, experiment_series, active_fill=False, min_samples=2):
    dfs = kg.prepare_dfs_of_experiments(knowledge_graph, experiment_series,
                                        fill_nones=active_fill)  # get user given insights and changed process parameters from the given graph
    filled_dfs = {key: kg.fill_nans_of_clustering_df(graph=knowledge_graph, df=df, type=key, active_fill=active_fill)
                  for key, df in dfs.items()}
    # TODO do it before filling or after?! -> after filling since we might get more values but do not encode default fill
    # InfluenceVisualiser.__enocde_categorical_values(filled_dfs)
    filled_dfs = {key: filled_df.fillna(-5) for key, filled_df in filled_dfs.items()}

    # append the key (e.g. 'influence') so we do not have column collissions/overwriting in pandas.concat
    for key, df in filled_dfs.items():
        renaming_dict = {name: key + '-' + name for name in df.columns}
        filled_dfs[key] = df.rename(renaming_dict, axis=1)

    # create lookup options to determine whether clustering was usefull - i.e. whether parametrisation assigned the same param cluster also exhibit similar clusterings for quality
    # clusters_param, expt_to_cluster_param, cluster_to_exp_param = clustering(pandas.concat(
    #     [filled_dfs['changed_parameters'], filled_dfs['changed_parameters_relative'], filled_dfs['quality'],
    #      filled_dfs['quality_relative'], filled_dfs['influences'], filled_dfs['influences_relative']], axis=1,
    #     sort=False))
    dfselector_one = 'changed_parameters'
    dfselector_two = 'quality'
    clusters_param, expt_to_cluster_param, cluster_to_exp_param = clustering_visualisation(filled_dfs[dfselector_one], min_samples=min_samples, filename='figs/clustering_'+dfselector_one)
    cluster_qual, exp_to_cluster_qual, cluster_to_exp_qual = clustering_visualisation(filled_dfs[dfselector_two], min_samples=min_samples, filename='figs/clustering_'+dfselector_two)


    amount_exp_in_qual_cluster = defaultdict(None)
    for cluster_param in clusters_param:
        exp_to_qual_cluster = defaultdict(int)
        # exp_to_qual_cluster == exp_to_cluster_qual?!
        for exp_id in cluster_to_exp_param[cluster_param]:
            exp_to_qual_cluster[exp_id] = exp_to_cluster_qual[exp_id]
        qual_cluster_to_exps_in_cluster = defaultdict(list)
        for exp, qual_cluster in exp_to_qual_cluster.items():
            qual_cluster_to_exps_in_cluster[qual_cluster].append(exp)

        # convert the results to a dictionary mapping qual_cluster to a list of experiments that are in the corresponding param_cluster
        ds = defaultdict(list)
        for qual_cluster,experiments in qual_cluster_to_exps_in_cluster.items():
            # appended to list with one element to make it easier to create dataframe
            ds[qual_cluster].append(len(experiments))
        df = pandas.DataFrame.from_dict(ds).sort_index(axis=1)
        # ax = df.plot.bar()
        # ax.set_title(cluster_param)
        # unwrap ds lists and add them to cluster matrix
        amount_exp_in_qual_cluster[cluster_param] = {qual_cluster:amount[0] for qual_cluster,amount in ds.items()}
        # plt.show()

    ds_qual = defaultdict(list)
    for cluster, experiments in cluster_to_exp_qual.items():
        ds_qual[cluster].append(len(experiments))
    df = pandas.DataFrame.from_dict(ds_qual).sort_index(axis=1)
    # ref_ax = df.plot.bar()
    # ref_ax.set_title("qual overall")
    # plt.show()

    matrix = pandas.DataFrame.from_dict(amount_exp_in_qual_cluster).sort_index(axis=1).sort_index(axis=0)
    mask = np.zeros_like(matrix)

    plt.figure(figsize=(20,20))
    ax = sns.heatmap(matrix, mask=mask, annot=True, square=True, fmt='g', cbar=False)
    ax.set(xlabel='Parameter ClusterID', ylabel='Quality ClusterID')
    plt.savefig(fname=f'figs/clusteringmatrix_{dfselector_one}_{dfselector_two}.pdf')
    plt.show()

    # dict = defaultdict(None)
    # for param_cluster, qual_clusters in amount_exp_in_qual_cluster.items():
    # [param_cluster [qual_cluster, count for qual_cluster, count in qual_clusters] for param_cluster, qual_clusters in amount_exp_in_qual_cluster]


def clustering_visualisation(df, calc_tsne=False, min_samples = 2, onedim_plot=True, filename=None):
    normed_df = pandas.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
    from scipy.spatial.distance import minkowski
    labels = OPTICS(min_samples=min_samples, metric=minkowski).fit(normed_df).labels_
    onedim_labels = [-1 if label == -1 else 1 for label in labels]
    clusters_per_experiments = dict(zip(normed_df.index, labels))
    cluster_to_exp = defaultdict(list)
    for exp_id,cluster in clusters_per_experiments.items():
        cluster_to_exp[cluster].append(exp_id)
    elements_in_cluster = defaultdict(int)
    for c in clusters_per_experiments.items():
        elements_in_cluster[c[1]] = elements_in_cluster[c[1]] + 1
    if calc_tsne:
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=600)
        tsne_results = tsne.fit_transform(normed_df)
        res = pandas.DataFrame()
        res['x'] = tsne_results[:, 0]
        res['y'] = tsne_results[:, 1]
        if onedim_labels:
            res['clusters'] = onedim_labels
            plt.figure()
            g = sns.scatterplot(
                x="x", y="y",
                style="clusters",
                hue="clusters",
                data=res
            )
            g.set(xlabel=None)
            g.set(ylabel=None)
            g.set(xticklabels=[])
            g.set(yticklabels=[])
            g.tick_params(bottom=False)
            g.tick_params(left=False)
            g.legend_.remove()
        else:
            res['clusters'] = labels
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x="x", y="y",
                hue="clusters",
                data=res,
                legend="full",
                alpha=0.3
            )
        if filename:
            plt.savefig(filename+'.pdf')
        plt.show()

    return set(labels), clusters_per_experiments, cluster_to_exp


if __name__ == "__main__":
    from data_provider.aipe_data_provider import AipeDataProvider

    dp = AipeDataProvider()
    viz = InfluenceVisualiser()

    data, lok, lov, boolean_parameters, returned_graphs, experiment_series, _ = dp.get_experiments_with_graphs(
        influential_influences_only=True)
    # data_all_vert, lok_all_vert, lov_all_vert, gs_all_vert, experiment_series_all_vert = data_provider.get_executed_experiments_data(
    #     completed_only=False, labelable_only=False, containing_insights_only=True, include_oneoffs=True,
    #     influential_only_in_graph=False)
    # knowledge_graph_all_vert = kg.aggregate(gs_all_vert,all_influences=True)
    # viz.plot_graph(knowledge_graph_all_vert,
    #                fname='figs/aggregated_all_vert_v' + str(len(knowledge_graph_all_vert.vs)) + '_e' + str(
    #                    len(knowledge_graph_all_vert.es)) + '.pdf')
    import json

    with open('experiment_series.json', 'w') as f:
        json.dump(experiment_series, f)

    knowledge_graph = kg.aggregate_unfiltered(returned_graphs)
    filename = 'figs/aggregated_v' + str(len(knowledge_graph.vs)) + '_e' + str(len(knowledge_graph.es)) + '.pdf'
    #create_directories_if_necessary(filename)
    #viz.plot_graph(knowledge_graph, fname=filename)
    # lower_quartile = lambda weights: plt.boxplot(weights)['boxes'][0].get_ydata()[0]
    # filtered_graphs = viz.clustering_experiment_level(knowledge_graph, experiment_series, threshold_function=lower_quartile, active_fill=False)
    # for key, filtered_graph in filtered_graphs.items():
    #     filename = 'figs/dbscan/' + key + '_v_' + str(len(filtered_graph.vs)) + '_e' + str(
    #         len(filtered_graph.es)) + '.pdf'
    #     viz.plot_graph(filtered_graph, fname=filename, title=key)
#    filtered_graphs = viz.weighting_relations_level(knowledge_graph, experiment_series,
#                                                      df_computation_fct=lower_quartile, active_fill=False)
#    for key, filtered_graph in filtered_graphs.items():
#        filename = 'figs/dbscan/' + key + '_v_' + str(len(filtered_graph.vs)) + '_e' + str(
#            len(filtered_graph.es)) + '.pdf'
#        viz.plot_graph(filtered_graph, fname=filename, title=key)

    # viz.plot_all_envfactor_parameter_pairs_relative(knowledge_graph, experiment_series)

    cluster_quality_visualisation(knowledge_graph, experiment_series, True, min_samples=5)



