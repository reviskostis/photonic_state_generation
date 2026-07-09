import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os
import random
from typing import Any

# get the parent directory of the folder containing the Python file
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)


class GraphstateGenerator(object):

    def __init__(self, nodes: list[int], edges: list[tuple]):
        nodes = np.array(nodes, dtype=int)
        edges = np.array(edges)
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        # catch graphs initialized from 0...
        # if min(nodes) == 0:
        #     nodes += 1
        #     edges += 1

        self.graph = nx.Graph()
        # self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    def plot(self, pos=None, options={}, show=True, name=None, coloring=None):
        if coloring == None:
            nx.draw(self.graph, pos=pos, with_labels=True, **options)
        else:
            if pos == None:
                pos = nx.kamada_kawai_layout(self.graph)
            colors = np.array(list(mcolors.TABLEAU_COLORS.keys()))
            for c_ind, nodelist in enumerate(coloring):
                nx.draw_networkx_nodes(self.graph, pos, nodelist=nodelist,
                                       node_color=colors[c_ind], **options, label="emitter "+str(c_ind+1))

            nx.draw_networkx_edges(self.graph, pos)
            nx.draw_networkx_labels(
                self.graph, pos)

            legend = plt.legend(scatterpoints=1, fontsize=10)
            for handle in legend.legend_handles:
                handle.set_sizes([30.])

        ax = plt.gca()
        ax.margins(0.2)
        plt.axis("off")

        if name is not None:
            plt.savefig(f"{name}.svg", format="SVG")
        if show:
            plt.show()
        pass

    def get_edges(self):
        return self.graph.edges()

    def get_nodes(self):
        return self.graph.nodes()

    # @staticmethod
    # def get_graph_from_adj(adjacency_matrix):
    #     n = len(adjacency_matrix)
    #     edges = np.stack(np.where(adjacency_matrix == 1)).T
    #     g = GraphstateGenerator(np.arange(n), edges)
    #     return g

    """Alternatively just call nx.adjacency_matrix(G, nodelist=ordering)"""
    @staticmethod
    def permute_adjacency_matrix(adjacency_matrix, reorder):
        """Swaps two (or more) vertices in the adjacency matrix from original order (1,2,3,...)
            to reorder(3,1,2,...)

        Arguments:
            adjacency_matrix -- adjacency matrix of an undirected graph
            reorderings -- list of vertex permutation

        Returns:
            one (or multiple) reordered adjacency matrices
        """
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("The adjacency matrix must be a square matrix.")

        n_nodes = adjacency_matrix.shape[0]
        permuted_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        permuted_matrix = adjacency_matrix[reorder][:, reorder]

        return permuted_matrix

    @staticmethod
    def get_tableau_from_adj(adjacency_matrix):
        n = len(adjacency_matrix)
        X_tab = np.eye(n, dtype=bool)
        tableau = np.hstack((X_tab, adjacency_matrix.astype(bool)), dtype=bool)
        return tableau

    @staticmethod
    def local_complementation(input_graph: nx.Graph, node):
        graph = input_graph.copy()
        sub_graph = nx.ego_graph(graph, node, 1, center=False)
        graph = nx.compose(graph, nx.complement(sub_graph))
        graph.remove_edges_from(sub_graph.edges())
        return graph

    @staticmethod
    def _LA_cost(graph, arrangement):
        cost = 0
        for edge in graph.edges():
            u, v = edge
            u_i = arrangement.index(u)
            v_i = arrangement.index(v)
            cost += np.abs(u_i - v_i)
        return cost

    @staticmethod
    def minLa_sim_anneal(graph, initial_arrangement, temperature, cooling_rate, iterations):
        current_arrangement = initial_arrangement.copy()
        current_cost = GraphstateGenerator._LA_cost(graph, current_arrangement)

        for _ in range(iterations):
            next_arrangement = current_arrangement.copy()
            i, j = random.sample(range(len(next_arrangement)), 2)
            next_arrangement[i], next_arrangement[j] = next_arrangement[j], next_arrangement[i]

            next_cost = GraphstateGenerator._LA_cost(graph, next_arrangement)

            if next_cost < current_cost or random.random() < np.exp((current_cost - next_cost)/temperature):
                current_arrangement = next_arrangement
                current_cost = next_cost

            temperature *= cooling_rate

        return current_arrangement, current_cost

    @staticmethod
    def leaf_contraction(graph: nx.Graph):

        leaves = [node for node in graph.nodes() if graph.degree(node) == 1]

        contracted_nodes = {}

        for leaf in leaves:
            neighbor = list(graph.neighbors(leaf))[0]

            graph = nx.contracted_nodes(
                graph, neighbor, leaf, self_loops=False)

            contracted_nodes[neighbor] = leaf

        return graph, contracted_nodes

    # old LC method

    # @staticmethod
    # def local_complementation(input_graph: nx.Graph, node):
    #     graph = input_graph.copy()
    #     neighbors = list(nx.all_neighbors(graph, node))
    #     subgraph = graph.subgraph([node, *neighbors])
    #     subgraphedges = list(subgraph.edges())
    #     subgraphcomplement = nx.complement(subgraph)

    #     # add neighbors of node
    #     subcompledges = subgraphcomplement.edges()
    #     graph.add_edges_from(subcompledges)
    #     for i in neighbors:
    #         subgraphcomplement.add_edge(node, i)
    #     subcompledges = list(subgraphcomplement.edges())
    #     for i in subgraphedges:
    #         if i not in subcompledges:
    #             graph.remove_edge(i[0], i[1])

    #     complement_graph = graph.copy()  # GraphstateGenerator(graph.nodes(), graph.edges())
    #     return complement_graph

    pass
