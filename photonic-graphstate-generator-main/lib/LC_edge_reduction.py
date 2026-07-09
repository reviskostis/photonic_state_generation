from lib.generate_graph import *
import networkx as nx
import numpy as np
import sys
import time
sys.setrecursionlimit(10000)


class LCReduction(object):
    """This class implements an edge reduction technique for graphs using local complementations.
        For comparison there is a variant from Ghanbari et al. arXiv:2401.00635 also implemented in their GraphiQ software"""

    def __init__(self, graph: nx.Graph, **kwargs):
        self.flag = False
        self.temp_path = []
        self.paths_result = []
        self.path_min_edges = []
        self.inc_path = []
        self.num_edges = []
        self.input_graph = graph.copy()
        self.increase_path = []
        self.reduced_graph = nx.Graph()
        self.temp_graph = graph.copy()
        self.run = False
        self.Ghan_prev_node = None

    def _explore_complementations(self, graph: nx.Graph, exhaustive: bool = False, branching=1, graph_blas=False):
        """Helperfunction: explore the nodes of the graphs to find a good candidate one to perform a local complementation on

        Arguments:
            graph -- input networkx graph

        Keyword Arguments:
            exhaustive -- if True returns all nodes that have a clustering coefficient larger than 0.5 (default: {False})
            branching -- int defines the number of nodes to return if multiple nodes are candidates  (default: {1})
            graph_blas -- If True uses the graph_blas library to calculate clustering coefficient more efficiently for large graphs (default: {False})

        Returns:
            a list of nodes on which local complementation would reduce the number of edges.
        """

        self.run = True
        if graph_blas:
            import graphblas_algorithms as ga
            sparse_graph = ga.Graph.from_networkx(graph)
            clustering = dict(nx.clustering(sparse_graph))
        else:
            clustering = dict(nx.clustering(graph))
        clustering_coeffs = np.array(list(clustering.values()))
        dict_nodes = np.array(list(clustering.keys()))

        possible_nodes = dict_nodes[np.where(clustering_coeffs > 0.5)[0]]
        possible_paths = []
        if len(possible_nodes) == 0:
            self.path_min_edges.append(graph.number_of_edges())
            return []
        elif exhaustive:
            return possible_nodes

        weighted_coeffs = [
            nx.degree(graph, node) for node in possible_nodes]  # *clustering[node]

        possible_paths = possible_nodes[np.where(
            weighted_coeffs == np.max(weighted_coeffs))[0]]

        if type(possible_paths) == np.ndarray:
            if len(possible_paths) < branching:
                possible_paths = list(possible_paths)
            else:
                possible_paths = list(
                    np.random.choice(possible_paths, branching, replace=False))
        elif type(possible_paths) == np.int64:
            possible_paths = [possible_paths]

        return possible_paths

    def edge_reduction(self, graph: nx.Graph = None, with_increase: bool = False, exhaustive: bool = False, branching=1, graph_blas=False):
        """This is the main function to reduce the number of edges

        Keyword Arguments:
            graph -- input networkx Graph (default: {None})
            with_increase -- Uses an edge increase function first before finding a minimum (default: {False})
            exhaustive -- ONLY for small size graph to explore all directions that decreases the number of edges without checking for isomorphisms (default: {False})
            branching -- an intermediate function to allow more reduction path but already might increase runtime significantly (default: {1})
            graph_blas -- If True uses the graph_blas library to calculate clustering coefficient more efficiently for large graphs (default: {False})

        Returns:
            a list of edge increasing LC-paths (depends how many paths are explored.)
        """
        if graph_blas:
            import graphblas_algorithms as ga

        if graph == None:
            temp_graph = self.input_graph.copy()
        else:
            temp_graph = graph.copy()

        if with_increase:
            self.increase_path = self.edge_increase(
                temp_graph, max_recursions=1000)
            for LC in self.increase_path:
                temp_graph = GraphstateGenerator.local_complementation(
                    temp_graph, LC)

        paths = []
        possible_paths = self._explore_complementations(
            temp_graph, exhaustive=exhaustive, branching=branching)
        if len(possible_paths) == 0:
            self.num_edges.append(temp_graph.number_of_edges())
            pass
        for node in possible_paths:
            paths.append(node)
            self.temp_path.append(node)
            next_graph = GraphstateGenerator.local_complementation(
                temp_graph.copy(), node)

            next_path = self.edge_reduction(
                graph=next_graph, exhaustive=exhaustive, branching=branching)

            if next_path:
                paths.append(next_path)
            if not next_path:
                self.paths_result.append(self.temp_path)
                self.temp_path = self.temp_path[:-1]
            if node == possible_paths[-1]:
                self.temp_path = self.temp_path[:-1]

        return paths

    def edge_increase(self, graph: nx.Graph, recursions=0, max_recursions=30, graph_blas=False):
        """Increasing the edges of a graph to probably escape a local minimum

        Arguments:
            graph -- networkx Graph

        Keyword Arguments:
            recursions -- an internal value to track recursions (default: {0})
            max_recursions -- int maximum allowed recursion depth (default: {30})
            graph_blas -- If True uses the graph_blas library to calculate clustering coefficient more efficiently for large graphs (default: {False})

        Returns:
            A list of local complementations from the input graph to the one with increased edges.
        """
        temp_graph = graph.copy()
        if graph_blas:
            import graphblas_algorithms as ga
            sparse_graph = ga.Graph.from_networkx(temp_graph)
            clustering = dict(nx.clustering(sparse_graph))
        else:
            clustering = dict(nx.clustering(graph))

        clustering_coeffs = np.array(list(clustering.values()))
        nodes = np.array(list(clustering.keys()))
        max_inc_nodes = nodes[np.where(
            np.logical_and(clustering_coeffs == np.min(clustering_coeffs), clustering_coeffs < 0.5))[0]]

        if len(max_inc_nodes) == 0 or recursions >= max_recursions:
            return self.inc_path
        if len(max_inc_nodes) > 1:
            degrees = np.array(temp_graph.degree(max_inc_nodes)).T
            max_degree = np.where(degrees[1] == np.max(degrees[1]))[0]
            max_inc_nodes = max_inc_nodes[max_degree]

        if len(max_inc_nodes) > 1:
            node = np.random.choice(max_inc_nodes)
        else:
            node = max_inc_nodes[0]
        if recursions == 0:
            self.inc_path = []
            self.inc_path.append(node)
            next_graph = GraphstateGenerator.local_complementation(
                temp_graph, node)
            recursions += 1
            self.edge_increase(graph=next_graph, recursions=recursions,
                               max_recursions=max_recursions)
        else:
            if node != self.inc_path[-1]:
                self.inc_path.append(node)
                next_graph = GraphstateGenerator.local_complementation(
                    temp_graph, node)
                recursions += 1
                self.edge_increase(
                    graph=next_graph, recursions=recursions, max_recursions=max_recursions)
        return self.inc_path

    def explore_complementations_Ghanbari(self, graph: nx.Graph, graph_blas=False):
        if graph_blas:
            import graphblas_algorithms as ga
            sparse_graph = ga.Graph.from_networkx(graph)
            clustering = dict(nx.clustering(sparse_graph))
        else:
            clustering = dict(nx.clustering(graph))
        clustering_coeffs = np.array(list(clustering.values()))
        dict_nodes = np.array(list(clustering.keys()))
        if np.all(clustering_coeffs <= 0.5):
            return []

        weighted_coeffs = [nx.degree(graph, node)*clustering[node]
                           for node in dict_nodes]

        sorted_ind = np.argsort(weighted_coeffs)
        # sorted_coeffs = weighted_coeffs[sorted_ind]
        possible_paths = dict_nodes[np.where(
            weighted_coeffs == np.max(weighted_coeffs))[0]]
        i = 1
        while True:
            if sorted_ind[-i] == self.Ghan_prev_node:
                i += 1
            else:
                possible_paths = [sorted_ind[-i]]
                self.Ghan_prev_node = sorted_ind[-i]
                return possible_paths

    def edge_reduction_Ghanbari(self, graph: nx.Graph = None, with_increase: bool = False, iter=0, max_iter=300):
        if graph == None:
            temp_graph = self.input_graph.copy()
        else:
            temp_graph = graph.copy()
        paths = []
        if with_increase:
            self.increase_path = self.edge_increase(
                temp_graph, max_recursions=1000)
            for LC in self.increase_path:
                temp_graph = GraphstateGenerator.local_complementation(
                    temp_graph, LC)

        possible_paths = self.explore_complementations_Ghanbari(
            temp_graph)
        if len(possible_paths) == 0 or iter > max_iter:
            self.num_edges.append(temp_graph.number_of_edges())
            pass

        for node in possible_paths:
            if iter < max_iter:
                self.Ghan_prev_node = possible_paths[0]
                paths.append(node)
                self.temp_path.append(node)
                next_graph = GraphstateGenerator.local_complementation(
                    temp_graph.copy(), node)

                next_path = self.edge_reduction_Ghanbari(
                    graph=next_graph, iter=iter+1, max_iter=max_iter)

                if next_path:
                    paths.append(next_path)
                    # self.path_min_edges.append(next_graph.number_of_edges())

                if not next_path:
                    self.paths_result.append(self.temp_path)
                    self.path_min_edges.append(next_graph.number_of_edges())
                    self.temp_path = self.temp_path[:-1]
                if node == possible_paths[-1]:
                    self.temp_path = self.temp_path[:-1]

        return paths

    def edge_reduction_Ghanbari2(self, graph: nx.Graph = None, with_increase: bool = False, iter=0, max_iter=300):
        if graph == None:
            temp_graph = self.input_graph.copy()
        else:
            temp_graph = graph.copy()
        if with_increase:
            self.increase_path = self.edge_increase(
                temp_graph, max_recursions=1000)
            for LC in self.increase_path:
                temp_graph = GraphstateGenerator.local_complementation(
                    temp_graph, LC)

        paths = [i for i in self.increase_path]

        while True:
            possible_path = self.explore_complementations_Ghanbari(temp_graph)
            if possible_path:
                self.Ghan_prev_node = possible_path[0]
            if len(possible_path) == 0 or iter > max_iter:
                self.num_edges.append(temp_graph.number_of_edges())
                self.paths_result.append(paths)
                self.path_min_edges.append(temp_graph.number_of_edges())
                break
            paths.append(possible_path[0])
            temp_graph = GraphstateGenerator.local_complementation(
                temp_graph.copy(), possible_path[0])
            iter += 1

        return paths

    def get_path(self):
        """if multiple paths exists this function returns randomly one of the paths 
        that hence the least number of edges

        Raises:
            Exception: If no reduction algorithm is run first (no paths exists)

        Returns:
            one LC-path list of integers to perform local complementations on
        """
        if self.increase_path:
            path = self.increase_path
        else:
            path = []

        if len(self.paths_result) == 1:
            for node in self.paths_result[0]:
                path.append(node)
        elif not self.run:
            raise Exception("Run a reduction algorithm first")
        else:
            if self.paths_result:
                min_paths = np.where(self.path_min_edges ==
                                     np.min(self.path_min_edges))[0]
                for node in self.paths_result[np.random.choice(min_paths)]:
                    path.append(node)

        return path

    @staticmethod
    def _non_repeating_randoms(min, max):
        """Helperfunction to produce a random list of local complementations avoids double LC on the same node

        Arguments:
            min -- 0
            max -- number of nodes in the graph

        Yields:
            a 'random' number for local complementation
        """
        previous_random = None
        while True:
            current_random = np.random.randint(min, max)
            if current_random != previous_random:
                previous_random = current_random
                yield current_random

    @staticmethod
    def random_LC(input_graph: nx.Graph, num_LC):
        """Performs num_LC random Local complementations on the given graph

        Arguments:
            input_graph -- networkx graph
            num_LC -- number of random local complementations

        Returns:
            new networkx graph LC-equivalent to input graph
        """
        graph = input_graph.copy()
        random_gen = LCReduction._non_repeating_randoms(
            0, graph.number_of_nodes()-1)
        for _ in range(num_LC):
            next_random = next(random_gen)
            graph = LCReduction.local_complementation(
                graph, next_random)

        return graph

    @staticmethod
    def local_complementation(input_graph: nx.Graph, node):
        """local complementations on networkx graph

        Arguments:
            input_graph -- graphs
            node -- target node for the local complementation

        Returns:
            the locally complemented networkx graph
        """
        graph = input_graph.copy()
        sub_graph = nx.ego_graph(graph, node, 1, center=False)
        lc_graph = nx.compose(graph, nx.complement(sub_graph))
        lc_graph.remove_edges_from(sub_graph.edges())
        return lc_graph
