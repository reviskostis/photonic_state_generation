import os
import pytest
from lib.generate_graph import *
import stim
import networkx as nx
import numpy as np


class TestGraphClass:

    @pytest.mark.parametrize("num_nodes", np.random.randint(3, 25, size=(10)))
    def test_graph_creation(self, num_nodes):
        nodes = np.arange(1, num_nodes+1, 1)
        edges = [tuple(np.random.randint(1, num_nodes+1) for _ in range(2))
                 for _ in range(10)]
        state = GraphstateGenerator(nodes, edges)
        check_graph = nx.Graph()
        check_graph.add_nodes_from(nodes)
        check_graph.add_edges_from(edges)

        assert check_graph.nodes() == state.get_nodes()
        assert check_graph.edges() == state.get_edges()

    def test_get_graph_tableau(self):

        pass

    # def test_localcomplementation(self):
    #     num_nodes = 5
    #     completegraph = nx.complete_graph(num_nodes)
    #     complement_graph = GraphstateGenerator.localcomplementation(
    #         completegraph, num_nodes-1)
    #     star_graph = nx.star_graph(num_nodes-1)

    #     assert nx.is_isomorphic(star_graph, complement_graph)

    #     complement_graph = GraphstateGenerator.localcomplementation(
    #         complement_graph, 0)

    #     assert nx.is_isomorphic(
    #         complement_graph, completegraph)

    pass
