import networkx as nx 
from graph_utils import *
from networkx.utils.misc import graphs_equal
from itertools import combinations  
from networkx.algorithms.community import kernighan_lin_bisection
class Cluster:
    def __init__(self, graph: Graph, nodes: Set[Vertex], inherited: Set | None = None):
        self.graph = graph
        self.nodes = nodes
        assert self.nodes.issubset(set(self.graph.nodes()))
        self.cutrank = cut_rank(graph, nodes)
        self.inherited = inherited

    def __add__(self, other):
        """Merge two clusters into a new one. """
        if not isinstance(other, Cluster):
            return NotImplemented
        if not graphs_equal(self.graph, other.graph):
            raise(NotImplementedError)
        if not(self.nodes.isdisjoint(other.nodes)):
            raise(NotImplementedError)
        return Cluster(self.graph, self.nodes.union(other.nodes), inherited={self, other})
      
    def __str__(self):
        return f"Cluster: {self.nodes}"

    @property
    def size(self):
        """The number of vertices in the cluster."""
        return len(self.nodes)

    def get_inheritance(self):
        """Get the inheritance tree of this cluster."""
        if self.inherited is None:
            return self.nodes
        else:
            return [ child.get_inheritance() for child in self.inherited]
        pass


class Clustering:
    """Base class for clustering algorithms."""

    def do(self, graph: Graph):
        """Trivial clustering (put everything in one big cluster.)"""
        return Cluster(graph, set(graph.nodes))
    pass


class GreedyClustering(Clustering):
    """Greedy clustering algorithm that merges clusters to minimize cut rank."""
    def init_clusters(self, graph: Graph) -> List[Cluster]:
        """Initialize clusters with one vertex each."""
        initial_clusters = []
        for vertex in graph.nodes():
            initial_clusters.append(Cluster(graph, {vertex}))
        return initial_clusters

    def find_pairing(self, cluster_list: List[Cluster]):
        """Find a pair of clusters to merge to minimize cut rank."""
        min_cutrank = (cluster_list[0] + cluster_list[1]).cutrank
        min_pairing = [0, 1]
        max_size = (cluster_list[0] + cluster_list[1]).size
        for a,b in combinations(range(len(cluster_list)), 2):
            c = cluster_list[a] + cluster_list[b]
            if (c.cutrank < min_cutrank) or ((c.cutrank == min_cutrank) and max_size < c.size):
                min_pairing = [a, b]
                min_cutrank = c.cutrank
                max_size = c.size
        return min_pairing

    def do(self, graph):
        """Perform the greedy clustering on the given graph."""
        cluster_list = self.init_clusters(graph)
        while len(cluster_list) != 1:
            pair = self.find_pairing(cluster_list)
            cluster_list[pair[0]] += cluster_list[pair[1]]
            cluster_list.pop(pair[1])
        return cluster_list[0]



class KernighanLinClustering(Clustering):
    """Kernighan-Lin based clustering algorithm."""
    def do(self, graph: Graph) -> Cluster:
        root_cluster = Cluster(graph, set(graph.nodes()))
        if len(root_cluster.nodes) < 2:
            return root_cluster
        nodes_1, nodes_2 = kernighan_lin_bisection(graph)
        cluster_1 = self.do(graph.subgraph(nodes_1))
        cluster_2 = self.do(graph.subgraph(nodes_2))
        root_cluster.inherited = cluster_1, cluster_2
        return root_cluster
    