import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import eulerize, is_eulerian, eulerian_circuit
import matplotlib.pyplot as plt
from collections import Counter
import random
from lib.generate_graph import *
from lib.tableau import *
from lib.circuitSolver import *


def find_longest_path(graph) -> list:
    """Tries to find the longest path in a graph. (Using depths first search)
    This function might not cover all nodes in a graph but the 
    minimum vertex disjoint pathcover is a hard problem

    Arguments:
        graph -- networkx graph

    Returns:
        a list of nodes for the path in the graph
    """
    longest_path = []

    def dfs_path(node, current_path):
        nonlocal longest_path
        current_path.append(node)

        for neighbor in graph.neighbors(node):
            if neighbor not in current_path:
                dfs_path(neighbor, current_path.copy())

        if len(current_path) > len(longest_path):
            longest_path = current_path

    for start_node in graph.nodes():
        dfs_path(start_node, [])

    return longest_path


def group_neighbors(graph, vertex_list):
    neighbors = {}
    non_neighbors = {}

    for vertex in vertex_list:
        neighbors[vertex] = [v for v in graph.neighbors(
            vertex) if v in vertex_list]
        non_neighbors[vertex] = [v for v in vertex_list if v !=
                                 vertex and v not in neighbors[vertex]]

    return neighbors, non_neighbors


def find_smallest_eulerian_supergraph(G):
    """Finds new edges to make an eulerian path in the graph
    Does a matching of vertices in the graph that are odd degrees
    allowed to have odd degree for two nodes

    Arguments:
        G -- networkx graph

    Returns:
        graph with added edges, list of added edges
    """
    odd_degrees = [node for node, degree in G.degree() if degree % 2 == 1]
    degrees = [degree for node, degree in G.degree() if degree % 2 == 1]
    connected, not_connected = group_neighbors(G, odd_degrees)
    insert_edge_graph = nx.Graph(not_connected)
    matching = nx.max_weight_matching(insert_edge_graph)

    if len(odd_degrees) % 2 == 1:
        print("Warning: odd number of vertices with odd degrees")
    # else:
    #     matching.pop()
    supergraph = G.copy()

    # Connect pairs of odd-degree vertices with the minimum number of additional edges
    added_edges = []
    for i in matching:
        supergraph.add_edge(i[0], i[1], color='red')
        added_edges.append(i)
    # for i in np.arange(0, len(odd_degrees), 2):
    #     u, v = np.random.choice(odd_degrees, size=2, replace=False)#odd_degrees[i], odd_degrees[i + 1]
    #     if not supergraph.has_edge(u, v):
    #         supergraph.add_edge(u, v, color='red')  # Added edges in red
    #         added_edges.append((u, v))
    #         odd_degrees.remove(u)
    #         odd_degrees.remove(v)

    return supergraph, added_edges


def hierholzer_eulerian_path(G):
    eulerian_path = []
    stack = [next(iter(G.nodes))]

    while stack:
        current_node = stack[-1]

        # Find a neighbor of the current node with available edges
        for neighbor in G.neighbors(current_node):
            if G.edges[current_node, neighbor].get('visited', False):
                continue

            # Mark the edge as visited
            G.edges[current_node, neighbor]['visited'] = True

            # Move to the neighbor
            stack.append(neighbor)
            break
        else:
            # No available edges, backtrack
            eulerian_path.append(stack.pop())

    return eulerian_path[::-1]


def split_eulerian_path_by_added_edges(eulerian_path, added_edges):
    """Splits the eulerian path if it crosses an added edge

    Arguments:
        eulerian_path -- list of nodes along the path
        added_edges -- list of added edges in the supergraph

    Returns:
        linear paths that are only crossing edges of the original graph
    """
    linear_subgraphs = []
    current_subgraph = []

    for node in eulerian_path:
        if current_subgraph and (current_subgraph[-1], node) in added_edges or current_subgraph and (node, current_subgraph[-1]) in added_edges:
            # Save the current linear subgraph
            linear_subgraphs.append(current_subgraph)
            current_subgraph = [node]
        else:
            current_subgraph.append(node)

    # Add the last subgraph
    linear_subgraphs.append(current_subgraph)

    return linear_subgraphs
