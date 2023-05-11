"""This module provides a methology to extract a subgraph from a graph
"""

import glob
import os
from pathlib import Path
import sys
import igraph as iG
from typing import List
import copy

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config import (
    KnowledgeGraphGeneratorType,
)
from knowledge_infusion.graph_embeddings.embedding_types import EmbeddingType


class SubGraphExtractor:
    _graph: iG.Graph
    _hops: int
    _avoid_literals: int
    _kgtype = KnowledgeGraphGeneratorType
    _tails: List[int]

    def __init__(self, graph: iG.Graph, kg_type: KnowledgeGraphGeneratorType) -> None:

        # Mapping to decide based on the kg type how many hops to do
        mapping = {
            KnowledgeGraphGeneratorType.UNQUANTIFIED: 1,
            KnowledgeGraphGeneratorType.BASIC: 1,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT2: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT3: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT2: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT3: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_LITERAL: 1,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3_WITH_LITERAL: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT: 3,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT2: 3,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT3: 3,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT: 3,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT2: 3,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT3: 3,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_LITERAL: 1,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3: 2,
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3_WITH_LITERAL: 2,
        }

        # If representation contains literals, do not include them in the SubKG
        if "with_literal" in kg_type.value:
            self._avoid_literals = True
        else:
            self._avoid_literals = False

        self._hops = mapping[kg_type]
        self._kgtype = kg_type
        self._graph = graph
        self._tails = []

    def extract_subgraph_as_iGraph(
        self, quality: str, quality_quantification: float = None
    ) -> iG.Graph:
        """Extracts the subgraph for a specific quality

        Args:
            quality (str): quality for which the subgraph should be extracted
            quality_quantification (float, optional): quality quantification - Defaults to None.

        Returns:
            iG.Graph: The iGraph
        """

        start_node_id = self._graph.vs.find(name=quality).index

        if quality_quantification == None:
            edge_list, _ = self.get_sub_graph_for_quality(start_node_id)
        else:

            if (
                self._kgtype in [
                    KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_LITERAL,
                    KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT,
                    KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT2,
                    KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT3,
                    KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT,
                    KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT2,
                    KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT3
                ]
            ):
                edge_list = self._get_subgraph_for_quantified_condition(
                    start_node_id, quality_quantification
                )

            elif (
                self._kgtype == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3
                or self._kgtype
                == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3_WITH_LITERAL
            ):
                edge_list = self._get_subgraph_for_w3_quantified_condition(
                    start_node_id, quality_quantification
                )

            else:
                edge_list, _ = self.get_sub_graph_for_quality(start_node_id)
        return self._edge_list_to_graph(edge_list)

    def _edge_list_to_graph(self, edgelist: List[iG.Edge]) -> iG.Graph:
        """Converts a list of edges into a graph

        Args:
            edgelist (List[iG.Edge]): list of edges

        Returns:
            iG.Graph: Graph generated from list of edges
        """

        # make deepcopy of the graph and destroys all edges in the copy.
        graph = copy.deepcopy(self._graph)
        graph.delete_edges()

        # Add all the edges that are present in the subgraph. Copy the original
        # edge attributes into the copy
        for edge in edgelist:
            n_edge = graph.add_edge(edge.source, edge.target)
            for att in edge.attribute_names():
                n_edge[att] = edge[att]

        # Delete all the vertices in the graph that are not part of the subgraph
        # Edges that are not part of the subgraph have no connections.
        to_delete = []

        for vertex in graph.vs:
            if len(vertex.neighbors()) == 0:
                to_delete.append(vertex.index)

        graph.delete_vertices(to_delete)

        return graph

    def _get_subgraph_for_w3_quantified_condition(
        self, startnodeid: int, quality_quantification: float
    ):
        """Converts a graph in the w3 conform representation into a edgelist
        considering the relevance of edges based on the actual quality_quantification
        present

        Args:
            startnodeid (int): id of the node of the quality
            quality_quantification (float): value of the quality

        Returns:
            List[iGraph.Edge]: Edges in the subgraph
        """

        relevant_edges = []  # All the edges of the subgraph

        start_node = self._graph.vs.find(startnodeid)
        start_node_connections = start_node.all_edges()

        pq_nodes = []

        # The quality is connected to the pq-relation nodes
        for connection in start_node_connections:
            relevant_edges.append(connection)
            pq_nodes.append(self._graph.vs.find(connection.target))

        mu_nus = []

        # Each pq-relation is connected to a parameter and a mu-nu relation
        # specifying it.
        for pq in pq_nodes:
            pq_edges = pq.out_edges()

            for edge in pq_edges:
                if edge["weight"] == "implied parameter":
                    relevant_edges.append(edge)  # Parameter
                if edge["weight"] == "quantified by":
                    mu_nus.append(self._graph.vs.find(edge.target))  # mu-nu

        # Find highest condition boundary
        all_boundaries = []
        for mu_nu in mu_nus:
            for edge in mu_nu.out_edges():
                if edge["weight"] in ["starts at", "ends at"]:
                    all_boundaries.append(
                        float(self._graph.vs.find(
                            edge.target)["literal_value"])
                    )
        max_end = max(all_boundaries)

        for mu_nu in mu_nus:
            for edge in mu_nu.out_edges():
                # Find out the range in which this mu-nu-relations is valid
                if edge["weight"] == "starts at":
                    _start = float(self._graph.vs.find(
                        edge.target)["literal_value"])
                elif edge["weight"] == "ends at":
                    _end = float(self._graph.vs.find(
                        edge.target)["literal_value"])

            start = min(_start, _end)
            end = max(_start, _end)
            # If mu-nu relation is valid add it to the sub graph
            if not start <= quality_quantification < end:
                if end == max_end == quality_quantification:
                    pass
                else:
                    continue
            for edge in mu_nu.all_edges():
                if self._avoid_literals and edge["literal_included"] != "None":
                    continue
                relevant_edges.append(edge)

        return relevant_edges

    def _get_subgraph_for_quantified_condition(
        self, startnodeid: int, quality_quantification: float
    ):
        """Gets the edge list for quantified_conditions with respect to the concrete
        quantification.

        Args:
            startnodeid (int): the id of the quality to start from
            quality_quantification (float): the value of the quantification

        Returns:
            List[iGraph.Edge]: List of edges in the subgraph
        """
        relevant_edges = []

        start_node = self._graph.vs.find(startnodeid)
        start_node_connections = start_node.all_edges()

        quantified_conclusions = []

        # qualities are either connected to the parameter or a mu
        for edge in start_node_connections:
            if edge["weight"] == "implies":
                # Directly add shortcut relations to the subkg
                relevant_edges.append(edge)
            elif edge["weight"] == "quantified by":
                # Save mus for further evaluation
                quantified_conclusions.append(edge)

        # Find highest condition boundary
        all_boundaries = []
        for edge in quantified_conclusions:
            target = edge.target
            target_vertex = self._graph.vs.find(target)
            target_edges = target_vertex.all_edges()
            for qc_edge in target_edges:
                if qc_edge["weight"] in ["starts at", "ends at"]:
                    all_boundaries.append(
                        float(self._graph.vs.find(
                            qc_edge.target)["literal_value"])
                    )
        max_end = max(all_boundaries)

        quantified_conclusions_in_correct_range = []
        for edge in quantified_conclusions:
            target = edge.target
            target_vertex = self._graph.vs.find(target)

            target_edges = target_vertex.all_edges()
            for qc_edge in target_edges:
                # Check if this is the right conclusion for the quality quantification
                if qc_edge["weight"] == "starts at":
                    _start = float(self._graph.vs.find(
                        qc_edge.target)["literal_value"])
                if qc_edge["weight"] == "ends at":
                    _end = float(self._graph.vs.find(
                        qc_edge.target)["literal_value"])
            start = min(_start, _end)
            end = max(_start, _end)
            if start <= quality_quantification < end or \
               end == max_end == quality_quantification:
                quantified_conclusions_in_correct_range.append(edge)

        # Add the conclusions, who are in the right range to the subgraph
        for edge in quantified_conclusions_in_correct_range:
            mu = self._graph.vs.find(edge.target)
            mu_edges = mu.all_edges()

            for mu_edge in mu_edges:
                if mu_edge["weight"] == "quantified by":
                    relevant_edges.append(mu_edge)
                if not self._avoid_literals:
                    if mu_edge['weight'] != "quantified by":
                        relevant_edges.append(mu_edge)
                    if mu_edge["weight"] == "implies":
                        nu = self._graph.vs.find(mu_edge.target)

                        ins = nu.all_edges()
                        for in_edge in ins:
                            if any(rel in in_edge["weight"] for rel in ["quantified by", "quantifies"]):
                                relevant_edges.append(in_edge)
        return relevant_edges

    def get_sub_graph_for_quality(self, start_node_id: int) -> (List[iG.Edge], List[iG.Vertex]):
        """
        Gets the edge list of the sub graph

        Args:
            startnodeid (int): the id of the quality to start from

        Returns:
            List[iGraph.Edge]: List of edges in the subgraph
            List[iGraph.Vertex]: List of vertecies in the subgraph
        """

        # First Step:
        #     This step does the propagation.
        #     This uses the hop-array as a state variable and the relevant_vertices-array as the output
        #     for every hop:
        #         1. Explore all neighbors of current hop
        #         2. Add neighbors to next hop
        #         3. Add newly found vertices in the current hop to the relevant_vertices
        #         4. current_hop = next hop

        relevant_vertices = set()  # Set for all vertices in the subgraph
        hop = [[start_node_id]]  # Nodes explored in each
        relevant_vertices.add(start_node_id)

        # Do the hops
        for current_hop in range(self._hops):
            current = []
            for element in hop[
                current_hop
            ]:  # For every vertex at the current hop stage
                # If parameter has been found stop hopping on this branch
                if self._graph.vs[element]["type"] == "parameter":
                    continue
                nbs = self._graph.vs[
                    element
                ].neighbors()  # Get the neighbors of the vertex
                for nb in nbs:
                    if self._avoid_literals:
                        if nb["type"] == "value":
                            continue

                    current.append(nb.index)  # Save neighbors
                    if nb["type"] == "parameter":
                        self._tails.append(nb.index)
            hop.append(current)  # Next hop = found neighbors
            relevant_vertices.update(current)  # Remember all vertices explored

        """
        Second Step:
            Find the relations between the vertices in the subgraph.
            Because this is a directed Graph we can use the directions to make
            sure every edge is uniquely explored, by only using outgoing relations
        """
        # From all explored vertices get the outgoing edges
        relevant_edges = []
        for element in relevant_vertices:
            try:
                out_edges = self._graph.vs[element].out_edges()
            except TypeError:
                out_edges = self._graph.vs[int(element)].out_edges()
            for edge in out_edges:
                # Check if target of edge is in Subgraph
                if edge.target in relevant_vertices:
                    relevant_edges.append(edge)
        return relevant_edges, relevant_vertices

    def get_tails(self):
        return self._tails


if __name__ == "__main__":
    """This modules main is only expected to be run in a debugging context. Please
    only execute this if you need a graphical representation of the generated 
    subgraphs.
    """
    import warnings
    warnings.warn("This main method is for debug purposes only")

    # Which representation to display
    repr = "quantified_parameters_without_shortcut"
    aggregation_method = "groundtruth"
    import pickle

    base_path = f"knowledge_infusion/compare_methods/results/*/*/*/*/iteration0/"
    exps = glob.glob(base_path)
    for data_ext_path in exps:
        for repr in KnowledgeGraphGeneratorType:
            for emb in EmbeddingType:
                try:
                    edges_str = glob.glob(
                        f"{data_ext_path}{repr}_using_{emb}/embedding/graph_embeddings/edges_*.pkl")
                    verts_str = glob.glob(
                        f"{data_ext_path}{repr}_using_{emb}/embedding/graph_embeddings/verts*.pkl")
                    if len(edges_str) > 1 or len(verts_str) > 1:
                        raise ValueError("too many edges/vert files")
                    if len(edges_str) == 0 or len(verts_str) == 0:
                        continue
                    with open(edges_str[0], "rb") as in_f:
                        edges = pickle.load(in_f)
                    with open(verts_str[0], "rb") as in_f:
                        verts = pickle.load(in_f)
                except FileNotFoundError as e:
                    print(f"{e} not found - was {emb} run?")

                try:
                    if edges.empty:
                        raise ValueError(f"edges {edges} is empty")
                    if verts.empty:
                        raise ValueError(f"vertices {edges} is empty")
                    g = iG.Graph.DataFrame(
                        edges, directed=True, vertices=verts)
                    sge = SubGraphExtractor(
                        g, KnowledgeGraphGeneratorType(repr)
                    )

                    subgraph = sge.extract_subgraph_as_iGraph("overall_ok")
                except TypeError as e:
                    print(f"{e} at {data_ext_path}/{repr}/{emb}",
                          file=sys.stderr)
                except Exception as e:
                    print(f"{e} at {data_ext_path}/{repr}/{emb}",
                          file=sys.stderr)
                    continue
                if not os.path.exists(f"{str(Path(data_ext_path).parents[0])}/graphs"):
                    os.makedirs(
                        f"{str(Path(data_ext_path).parents[0])}/graphs")
                subgraph.vs["label"] = subgraph.vs["name"]
                import matplotlib.pyplot as plt

                layout = subgraph.layout("kk")
                fig, ax = plt.subplots()

 # None equals pq-rel bc setting the type was forgotten in qp_w3, otherwise it's always set
                color_dict = {"qual_influence": "blue",
                              "env_influence": "brown", "parameter": "yellow",
                              "value": "cyan", "pq-relation": "magenta", "quantified_conclusion": "teal", "my-ny-relation": "forestgreen", None: "magenta"}

                visual_style = {}
                try:
                    visual_style["vertex_color"] = [color_dict[type]
                                                    for type in subgraph.vs["type"]]
                except KeyError as e:
                    print("color_dict doesn't include " + str(e))
                visual_style["edge_label_size"] = 10  # [2,2,2]
                visual_style["margin"] = 80
                visual_style["vertex_size"] = 0.1
                visual_style['edge_label'] = subgraph.es['weight']

                iG.plot(subgraph, layout=layout, target=ax, **visual_style)

                plt.savefig(
                    f"{str(Path(data_ext_path).parents[0])}/graphs/{repr}_{emb}_rel.pdf")
                plt.clf()
                plt.cla()

                visual_style = {}
                try:
                    visual_style["vertex_color"] = [color_dict[type]
                                                    for type in g.vs["type"]]
                except KeyError as e:
                    print("color_dict doesn't include " + str(e))
                visual_style["edge_width"] = 0.1
                visual_style["vertex_size"] = 0.5
                #visual_style["layout"] = graph.layout_kamada_kawai()

                #    visual_style["bbox"] = (1500, 1500)
                #   visual_style["margin"] = 100

                fig, ax = plt.subplots()
                iG.plot(g, target=ax, **visual_style)
                plt.savefig(
                    f"{str(Path(data_ext_path).parents[0])}/graphs/{repr}_{emb}.pdf")
                plt.clf()
                plt.cla()
