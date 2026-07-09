# The objective of this file is to store vertex and edges of defined graphs

import numpy as np
import sys
import os
import networkx as nx

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)


def ring_graph(ordering):
    nodes = np.arange(len(ordering))
    edges = np.array([(ordering[i], ordering[(i+1) % len(ordering)])
                      for i in range(len(ordering))])
    return nodes, edges


def nine_q_shor(choice):
    n = 9
    nodes = np.arange(1, n+1, 1)
    match choice:
        case 1:
            edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3),
                     (2, 6), (2, 7), (3, 8), (3, 9)]
        case 2:
            edges = [(1, 3), (2, 3), (3, 6), (3, 9), (4, 6),
                     (5, 6), (6, 9), (7, 9), (8, 9)]
        case 3:
            edges = [(1, 2), (2, 3), (2, 5), (2, 8), (4, 5),
                     (5, 6), (5, 8), (7, 8), (8, 9)]
    return nodes, edges


def reduced_repeater_graph():
    n = 12
    nodes = np.arange(1, n+1, 1)
    # edges = [(1, 2), (2, 6), (2, 8), (2, 10), (2, 12), (3, 4), (4, 8), (4, 10),
    #         (4, 12), (5, 6), (6, 8), (6, 10), (6, 12), (7, 8), (8, 12), (9, 10), (11, 12)]
    # edges = [(1, 2), (2, 6), (2, 8), (2, 4), (2, 12), (3, 4), (10, 8), (4, 10),
    #         (10, 12), (5, 6), (6, 8), (6, 4), (6, 12), (7, 8), (8, 12), (9, 10), (11, 12)]
    edges = [(1, 2), (2, 4), (2, 6), (2, 8), (2, 12), (3, 4), (4, 6), (4, 8), (4, 12),
             (5, 6), (6, 8), (6, 10), (7, 8), (8, 10), (9, 10), (10, 12), (11, 12)]
    return nodes, edges


def ring_resource_qpc(num_encoded_qubits, n, m, p_q_ordering):
    num_qubits = n*m*num_encoded_qubits
    nodes = np.arange(0, num_qubits, 1)
    edges = []
    assert len(p_q_ordering) == n*m
    for i in range(num_encoded_qubits):
        for j in range(n):
            node_crossing_edges = [
                (nodes[(i*n*m+p_q_ordering[j*m+m-1]-1) % num_qubits], nodes[((i+1)*n*m+p_q_ordering[cr*m+m-1]-1) % num_qubits]) for cr in range(n)]
            encoding_edges = [
                (nodes[i*n*m+p_q_ordering[k+j*m]-1], nodes[i*n*m+p_q_ordering[k+1+j*m]-1]) for k in range(m-1)]

            edges.extend(encoding_edges)
            edges.extend(node_crossing_edges)
    return nodes, edges


def cube_graph(ordering):
    num_qubits = 8
    nodes = np.arange(1, num_qubits+1, 1)
    assert len(ordering) == num_qubits
    edges = np.array([(ordering[0], ordering[1]), (ordering[0], ordering[7]), (ordering[0], ordering[3]),
                      (ordering[2], ordering[3]), (ordering[2],
                                                   ordering[1]), (ordering[2], ordering[5]),
                      (ordering[3], ordering[4]), (ordering[4],
                                                   ordering[7]), (ordering[4], ordering[5]),
                      (ordering[7], ordering[6]), (ordering[6], ordering[1]), (ordering[6], ordering[5])])
    return nodes, edges


def snowflake():
    nodes = np.arange(1, 30)
    edges = np.array([(1, 2), (2, 3), (2, 4), (4, 6), (5, 6), (6, 7), (4, 8),
                     (8, 12), (12, 10), (12, 14), (9,
                      10), (10, 11), (13, 14), (14, 15),
                     (8, 19), (19, 17), (19, 21), (16,
                      17), (17, 18), (20, 21), (21, 22),
                     (8, 26), (26, 24), (26, 28), (23, 24), (24, 25), (27, 28), (28, 29)])
    return nodes, edges


def snowflake2():
    nodes = np.arange(1, 30)
    edges = np.array([(2, 3), (3, 4), (3, 5), (5, 7), (6, 7), (7, 8), (5, 1),
                     (1, 12), (12, 10), (12, 14), (9,
                      10), (10, 11), (13, 14), (14, 15),
                     (1, 19), (19, 17), (19, 21), (16,
                      17), (17, 18), (20, 21), (21, 22),
                     (1, 26), (26, 24), (26, 28), (23, 24), (24, 25), (27, 28), (28, 29)])
    return nodes, edges


def snowflake3():
    nodes = np.arange(1, 30)
    edges = np.array([(1, 2), (2, 3), (2, 4), (4, 6), (5, 6), (6, 7), (4, 29),
                     (29, 11), (11, 9), (11, 13), (8,
                      9), (9, 10), (13, 12), (13, 14),
                     (29, 18), (18, 16), (18, 20), (15,
                      16), (16, 17), (19, 20), (20, 21),
                     (29, 25), (25, 23), (25, 27), (22, 23), (23, 24), (26, 27), (27, 28)])
    return nodes, edges


def square_grid(n=3):
    nodes = np.arange(0, n**2)
    edges = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 6),
             (3, 4), (4, 7), (4, 5), (5, 8), (6, 7), (7, 8)]
    # for row in np.arange(0, n+1):
    #     for column in np.arange(0, n):
    #         i = row+column
    #         edges.append((i, i+1))
    #         edges.append((i, n+i))
    return nodes, edges


class encodedGraphs(object):
    def __init__(self, core_graph: nx.Graph, logical_graph: nx.Graph, logical_node: int) -> None:
        self.core_graph = core_graph.copy()
        self.logical_graph = logical_graph.copy()
        self.logical_node = logical_node

        self.encoded_graph = self.get_encoded_graph(
            self.core_graph, self.logical_graph, self.logical_node)
        self.encoded_graph = self.get_graph_integer_node(self.encoded_graph)

        pass

    def find_other_node(self, edge: tuple, node: int) -> int:
        """Find the other node in an edge.
        edge = (node, other_node)
        return other_node

        Args:
            edge (tuple): edge 
            node (int): node

        Raises:
            ValueError: node not in edge

        Returns:
            _type_: other node
        """
        """"""
        if node != edge[0]:
            return edge[0]
        elif node != edge[1]:
            return edge[1]
        else:
            raise ValueError(f"node {node} not found in edge {edge}.")

    def get_encoded_graph(self, core_graph: nx.Graph, logical_graph: nx.Graph, logical_node: int) -> nx.Graph:
        """Create an encoded graph based on a core graph and a logical encoding with graph code.
        The core graph gives the global structure of the resulting graph but each node is replaced by a logical qubit
        encoded using a graph code.
        A graph code has a graph structure and two types of nodes, the physical qubits and the logical node.
        The logical node connectivity with its physical qubits gives the graph code structure.


        For the resulting graph, the nodes are named "(i, j)"
        with i the core node they belong to and j their label in the logical graph

        To build it we keep the connectivity between physical qubits in the same code.
        Each time a physical qubit is connected to its corresponding logical node,
        it is connected to other physical qubits from a different logical graph (different node from the core graph).
        The connectivity is such that neighbors of logical nodes from neighbor nodes (in the core graph) are connected together 


        Args:
            core_graph (nx.Graph): graph giving the global structure of the resulting graph
            logical_graph (nx.Graph): graph code used for the encoding
            node_logical (int): logical node in the graph code

        Returns:
            nx.Graph: resulting logically-encoded graph
        """
        encoded_graph = nx.Graph()

        # The logical node should correspond to a node in the logical graph.
        assert logical_node in logical_graph.nodes()

        # Neighbor list of the logical node within each logical graph.
        logical_neighbor_nodes = list(logical_graph.neighbors(logical_node))

        for core_node in core_graph.nodes():

            # Neighbors of the current node in the core_graph
            core_neighbors = list(core_graph.neighbors(core_node))

            for edge_logical in logical_graph.edges():
                if logical_node not in edge_logical:  # not an edge between involving the logical node
                    # Simply add it
                    encoded_graph.add_edge(
                        (core_node, edge_logical[0]), (core_node, edge_logical[1]))
                else:
                    node_phys = self.find_other_node(
                        edge_logical, logical_node)
                    # Add an edge between this node and the neighbors of the logical node in other core nodes
                    edges_to_add = [((core_node, node_phys), (node_1, node_2))
                                    for node_1 in core_neighbors for node_2 in logical_neighbor_nodes]
                    encoded_graph.add_edges_from(edges_to_add)

        return encoded_graph

    def find_index_node(self, nodes: np.array, node: tuple) -> int:
        """Find the index of a node in a node list.

        Args:
            nodes (np.array): node list
            node (tuple): node to find

        Returns:
            int: node's index in the node list
        """
        return np.where(np.all(nodes == np.array(node), axis=1))[0]

    def get_graph_integer_node(self, graph: nx.Graph) -> nx.Graph:
        """Return the same graph with nodes being labeled as integers
        ranging from 0 to graph.number_of_nodes() - 1.


        Args:
            graph (nx.Graph): input graph

        Returns:
            nx.Graph: same graph but with nodes labeled as integers
        """
        graph_nodes = np.array(graph.nodes())
        graph_nodes = graph_nodes[np.lexsort(
            (graph_nodes[:, 1], graph_nodes[:, 0]))]

        result_graph = nx.Graph()
        # Replace node indices in the new graph
        result_graph.add_edges_from(
            (
                self.find_index_node(graph_nodes, edge[0])[
                    0],  # First node index
                self.find_index_node(graph_nodes, edge[1])[
                    0]   # Second node index
            ) for edge in graph.edges()
        )

        return result_graph
