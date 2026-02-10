import time
from clustering import GreedyClustering, KernighanLinClustering
import networkx as nx
import matplotlib.pyplot as plt
from networkx.utils import reverse_cuthill_mckee_ordering
from optimization_script import run_optimization
import numpy as np
import glob
import os
import pickle
from clustering import Cluster
import itertools
import math
from optimization_script import run_optimization
from lib.circuitSolver import *
from lib.generate_graph import *
from lib.tableau import *
from clustering import GreedyClustering, KernighanLinClustering
import matplotlib.pyplot as plt
from clustering import Cluster

pauls_prop = False
use_database_state = False

def edge_reduction_func(input_graph):
    our_graph_original = input_graph.copy()
    # Create a mapping dictionary: (0, 0) -> 0, (0, 1) -> 1, ..., (2, 2) -> 8
    mapping = {node: i for i, node in enumerate(our_graph_original.nodes())}

    # Create a new graph with integer labels
    our_graph = nx.relabel_nodes(our_graph_original, mapping)
    graph , circ, statistics = run_optimization(our_graph, edge_reduction=True, minLA=False, opt_CNOT= False)
    return graph

def peel_to_rank1(cluster):
    """
    Recursively peel cluster hierarchy until all subclusters have cutrank=1.
    Returns list of cutrank-1 clusters.
    """
    if cluster.cutrank == 1:
        return [cluster]
    
    if cluster.inherited is None:
        # Base case: can't split further but cutrank > 1
        return [cluster]
    
    result = []
    for child in cluster.inherited:
        result.extend(peel_to_rank1(child))
    
    return result

def apply_ordering_to_clusters(graph, clusters):
    """
    Apply Reverse Cuthill-McKee ordering to each cluster.
    Generate an ordering (permutation) of the graph nodes to make a sparse matrix.
    Uses the reverse Cuthill-McKee heuristic (based on breadth-first search)
    """
    orderings = {}
    for cluster in clusters:
        subgraph = graph.subgraph(cluster.nodes)
        if len(cluster.nodes) > 0:
            try:
                # RCM returns a generator
                def biggest_degree(G):
                    return max(G, key=G.degree)
                ordering = list(reverse_cuthill_mckee_ordering(subgraph, heuristic=biggest_degree))
            except Exception as e:
                print(f"Error ordering cluster {cluster}: {e}")
                ordering = list(cluster.nodes)
            orderings[cluster] = ordering
        else:
            orderings[cluster] = []
    
    return orderings

def reconcile_orderings_with_bridges(graph, clusters, orderings):
    """
    Reconcile partial orderings considering bridge edges between clusters.
    """
    # Find inter-cluster edges (bridges)
    bridges = []
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters[i+1:], start=i+1):
            bridge_edges = [
                (u, v) for u in c1.nodes for v in c2.nodes 
                if graph.has_edge(u, v)
            ]
            if bridge_edges:
                bridges.append((i, j, bridge_edges))
    
    # Build constraint graph where clusters are nodes
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from(range(len(clusters)))
    for i, j, edges in bridges:
        cluster_graph.add_edge(i, j, weight=len(edges))
    
    # Find ordering of clusters
    # Simple DFS preorder as a heuristic for cluster arrangement
    if len(cluster_graph) > 0:
        cluster_ordering_indices = list(nx.dfs_preorder_nodes(cluster_graph, 0))
    else:
        cluster_ordering_indices = list(range(len(clusters)))
    
    # Concatenate based on cluster ordering
    global_ordering = []
    for idx in cluster_ordering_indices:
        global_ordering.extend(orderings[clusters[idx]])
    
    return global_ordering


#definde a functions to load graphs from the database of random graphs
def find_saved_graph(node_number, p, idx=None, search_root='.'):
    """Return path to a saved graph matching node_number and p (or None)."""
    p_safe = str(p)
    patterns = [
        f"**/*graph_n{node_number}_p{p_safe}_idx*.gpickle",
        f"**/*_graph_n{node_number}_p{p_safe}_idx*.gpickle",
        f"**/*p{p_safe}*/**/*graph_n{node_number}_p{p_safe}_idx*.gpickle",
        f"**/*p{p_safe}*/**/*_graph_n{node_number}_p{p_safe}_idx*.gpickle"
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(search_root, pat), recursive=True))
    matches = sorted(set(matches))
    if not matches:
        return None
    if idx is not None:
        candidates = [m for m in matches if f"_idx{idx}.gpickle" in m]
        if candidates:
            return candidates[0]
    return matches[0]
def load_graph_from_file(path):
    """Load NetworkX graph from a .gpickle or pickled file."""
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)



# START: new advanced approach
def count_active_cross_edges(graph, set_A_remaining, set_B_remaining):
    """
    Helper to count edges between A_remaining and B_remaining.
    (Optimized for clarity, can be cached for performance)
    """
    count = 0
    # This is the naive O(N^2) check. 
    # For strict Rank-1 cuts, this is simply len(connectors_A_in_rem) * len(connectors_B_in_rem)
    for u in set_A_remaining:
        for v in set_B_remaining:
            if graph.has_edge(u, v):
                count += 1
    return count

def dp_merge_orders(graph, order_A, order_B):
    """
    Merges two ordered lists of nodes (order_A, order_B) to minimize 
    Linear Arrangement cost of edges BETWEEN A and B.
    
    This is equivalent to finding the shortest path in a grid.
    """
    n = len(order_A)
    m = len(order_B)
    
    # DP State: dp[i][j] = min cumulative edge length using first i of A and j of B
    dp = np.full((n + 1, m + 1), np.inf)
    parent = {} # To reconstruct the path: (i, j) -> (prev_i, prev_j, 'A' or 'B')
    
    dp[0][0] = 0
    
    # Pre-calculate active edges for performance (Optional but recommended)
    # Ideally, we know the "connector" nodes. 
    # For now, we compute cost on the fly:
    # Cost added at step k = number of "open" edges crossing the cut.
    # Open edges = edges between (Placed_A, Remaining_B) and (Placed_B, Remaining_A)
    
    # We iterate through the grid
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            
            # Current sets of REMAINING nodes (not yet placed)
            # If we are at state (i, j), it means order_A[:i] and order_B[:j] are placed.
            # The "cost" of the current state is the number of active edges 
            # stretching across the future positions.
            
            rem_A = order_A[i:]
            rem_B = order_B[j:]
            placed_A = order_A[:i]
            placed_B = order_B[:j]
            
            # Calculate cross-cut active edges (this is the weight of being at state i,j)
            # Rank 1 Optimization: We only care about edges between A and B.
            # Edges internal to A or B are already fixed by recursive calls.
            
            # Note: This cost calculation is the "Cut Width" at this position.
            # Sum of Cut Widths = Total Edge Length.
            
            # Bridges stretching from Placed A -> Remaining B
            c1 = 0
            for u in placed_A:
                for v in rem_B:
                    if graph.has_edge(u, v): c1 += 1
            
            # Bridges stretching from Placed B -> Remaining A
            c2 = 0
            for u in placed_B:
                for v in rem_A:
                    if graph.has_edge(u, v): c2 += 1
            
            current_cut_cost = c1 + c2
            
            # Transitions
            # Coming from picking A[i-1] (Moving down)
            if i > 0:
                prev_cost = dp[i-1][j]
                if prev_cost + current_cut_cost < dp[i][j]:
                    dp[i][j] = prev_cost + current_cut_cost
                    parent[(i,j)] = (i-1, j, 'A')
            
            # Coming from picking B[j-1] (Moving right)
            if j > 0:
                prev_cost = dp[i][j-1]
                if prev_cost + current_cut_cost < dp[i][j]:
                    dp[i][j] = prev_cost + current_cut_cost
                    parent[(i,j)] = (i, j-1, 'B')

    # Backtrack to find the merge
    merged_order = []
    curr = (n, m)
    while curr != (0, 0):
        prev_i, prev_j, choice = parent[curr]
        if choice == 'A':
            # We picked A[i-1]
            merged_order.append(order_A[curr[0]-1])
        else:
            merged_order.append(order_B[curr[1]-1])
        curr = (prev_i, prev_j)
    
    return merged_order[::-1] # Reverse because we backtracked


def hierarchical_ordering(graph, cluster):
    """
    Recursive function replacing 'peel_to_rank1'.
    1. If cluster is leaf/rank1: Solve RCM.
    2. Else: Recurse children, then MERGE their orders.
    """
    # Base Case: Stop if cutrank is 1 OR no children
    if cluster.cutrank <= 1 or cluster.inherited is None:
        # Apply RCM to this leaf cluster
        if len(cluster.nodes) == 0: return []
        subgraph = graph.subgraph(cluster.nodes)
        try:
            # RCM
            return list(reverse_cuthill_mckee_ordering(subgraph))
        except:
            return list(cluster.nodes)
            
    # Recursive Step
    # Note: greedy clustering usually produces 2 children (binary tree)
    # But inherited is a list, so we fold them.
    
    child_orders = []
    for child in cluster.inherited:
        child_orders.append(hierarchical_ordering(graph, child))
    
    # Merge children sequentially (if > 2, merge 1 and 2, then result with 3...)
    final_order = child_orders[0]
    for next_order in child_orders[1:]:
        final_order = dp_merge_orders(graph, final_order, next_order)
        
    return final_order

# END: new advance approach

#-----------------------------------------------START: a new approach-----------------------------------------------


#-----------------------------------------------END: a new approach-----------------------------------------------



#-----------------------------------------------START: Brute Force-----------------------------------------------
# We know that this is not reasonable in general but it is good to have an idea on the true lower bound
def calculate_arrangement_cost(graph: nx.Graph, arrangement: tuple) -> int:
    """
    Calculates the total linear arrangement cost for a given permutation 
    of the graph's vertices.
    """
    # Create a mapping from vertex ID to its position (index) in the arrangement
    position_map = {node: pos for pos, node in enumerate(arrangement)}
    
    total_cost = 0
    
    # Iterate over all edges in the graph
    for u, v in graph.edges():
        # Edge length is the absolute difference in positions
        edge_length = abs(position_map[u] - position_map[v])
        total_cost += edge_length
        
    return total_cost

def min_linear_arrangement_brute_force(graph: nx.Graph) -> tuple:
    """
    Finds the Minimum Linear Arrangement (MinLA) using a brute-force approach,
    returning the optimal arrangement as a list (array).
    """
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    
    # Check for computational feasibility
    if num_nodes > 10:
        print(f"Graph has {num_nodes} nodes. There are {math.factorial(num_nodes):,} permutations.")
        print("Brute-force is computationally infeasible. Execution will be very long.")
        
    min_cost = float('inf')
    optimal_arrangement = None # Will store the optimal permutation as a tuple
    
    start_time = time.time()
    total_permutations = math.factorial(num_nodes)
    
    # Generate ALL possible permutations of the vertices
    for i, arrangement in enumerate(itertools.permutations(nodes)):
        
        current_cost = calculate_arrangement_cost(graph, arrangement)
        
        if current_cost < min_cost:
            min_cost = current_cost
            optimal_arrangement = arrangement
            
    end_time = time.time()
    
    print("\n--- Brute Force MinLA Results ---")
    print(f"Total nodes (n): {num_nodes}")
    print(f"Total permutations checked: {total_permutations:,}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print("---------------------------------")
    
    # Return the minimum cost and the optimal arrangement as a list (array)
    return min_cost, list(optimal_arrangement)

#-----------------------------------------------END: Brute Force-----------------------------------------------




#Load from the database

if use_database_state:
    node_number = 20
    p = 0.55
    idx = 7           # or set integer if you want a particular saved index
    search_root = '.'     # change to project root if running from another folder

    path = find_saved_graph(node_number, p, idx=idx, search_root=search_root)
    if path is None:
        raise FileNotFoundError(f"No saved graph found for n={node_number}, p={p} under {search_root}")

    graph = load_graph_from_file(path)
    print(f"Loaded graph: {path} -> nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

    adj = nx.to_numpy_array(graph, dtype=int)

    # print("Initial graph from random database")
    # nx.draw(graph, with_labels=True)
    # plt.show()
else:
    # Create a graph
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4,5), (5,6), (6,7), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (4,11)])



#previous optimizer
opt_graph, circuit, stats = run_optimization(graph, num_LA=10)

# print("\nNils: Optimized graph edges:", opt_graph.edges())
print("\nNils: opt emitters:", stats[0])
print("Nils: emission order:", stats[3])
print("Nils: CNOTS:", stats[2])
print("Nils: Arrangement cost:",calculate_arrangement_cost(graph, stats[3]))

# print("Print Ordering based on Nils")
# nx.draw(opt_graph, with_labels=True)
# plt.show()



if pauls_prop:
    # Method 1: Greedy clustering (minimizes cut rank)
    greedy = GreedyClustering()
    result_cluster = greedy.do(graph)
    print("Final cluster",result_cluster)  # Shows final cluster
    print("How it was built",result_cluster.get_inheritance())  # Shows how it was built

    # Method 2: Kernighan-Lin clustering (balanced bisection)
    kl = KernighanLinClustering()
    result_cluster = kl.do(graph)
    print("How it was built using Kernighan-Lin clustering",result_cluster.get_inheritance())  # Shows hierarchical tree


# # New logic: Peel to rank 1 and order
# print("\n--- New Logic: Peeling and Ordering ---")
# # We use the result from Greedy Clustering (which was stored in result_cluster earlier, but overwritten by KL)
# # Let's re-run Greedy to be sure
# greedy = GreedyClustering()
# root_cluster = greedy.do(graph)

# rank1_clusters = peel_to_rank1(root_cluster)
# print(f"Found {len(rank1_clusters)} clusters with cutrank=1")

# # for i, c in enumerate(rank1_clusters):
# #     print(f"  Cluster {i}: {c.nodes}, cutrank={c.cutrank}")

# orderings = apply_ordering_to_clusters(graph, rank1_clusters)

# # print("\nPartial orderings (Reverse Cuthill-McKee):")
# # for i, (cluster, ordering) in enumerate(orderings.items()):
# #     print(f"  Cluster {i}: {ordering}")

# global_ordering = reconcile_orderings_with_bridges(graph, rank1_clusters, orderings)
# print(f"\nCuthill-McKee opt emitters: {global_ordering}")


# # Visualize final ordering
# ordered_graph = nx.relabel_nodes(graph, {old: new for new, old in enumerate(global_ordering)})
# # nx.draw(ordered_graph, with_labels=True)
# # plt.show()


# opt_graph_ordered_graph, circuit, stats = run_optimization(ordered_graph, minLA=False)

# # print("Optimized graph edges:", opt_graph_ordered_graph.edges())
# print("Cuthill-McKee opt emitters:", stats[0])
# # print("Cuthill-McKee emission order:", stats[3])
# print("Cuthill-McKee CNOTS:", stats[2])

# # nx.draw(opt_graph_ordered_graph, with_labels=True)
# # plt.show()



print("\n--- Improved Logic: Hierarchical DP Merge ---")
#reduce edge
graph = edge_reduction_func(graph)
greedy = GreedyClustering()
root_cluster = greedy.do(graph)

# This single call does the peeling, leaf ordering, AND DP reconciliation
final_ordering_dp = hierarchical_ordering(graph, root_cluster)

print(f"Global Ordering (DP): {final_ordering_dp}")

# Visualize and Test
ordered_graph_dp = nx.relabel_nodes(graph, {old: new for new, old in enumerate(final_ordering_dp)})
opt_graph_dp, circuit, stats_dp_and_sa = run_optimization(ordered_graph_dp, num_LA= 10)

print("DP+SAminLA emitters:", stats_dp_and_sa[0])
print("DP+SAminLA CNOTS:", stats_dp_and_sa[2])
print("DP+SAminLA Arrangement cost:",calculate_arrangement_cost(graph, stats_dp_and_sa[3]))


