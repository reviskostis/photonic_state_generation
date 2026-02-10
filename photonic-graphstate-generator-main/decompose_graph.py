"""
Balanced Rank-1 Partitioning Problem Solver

This module implements algorithms for decomposing a graph into balanced partitions
where each partition has cut-rank 1 relative to the rest of the graph over GF(2).

Theory:
- A cut has rank-1 over GF(2) when vertices in Vi have identical neighborhoods in V\\Vi
- Such vertices form modules (or twin sets) in the graph
- Graphs fully decomposable via rank-1 cuts are Distance-Hereditary graphs

References:
- Oum & Seymour (2006): Rank-width and vertex-minors
- Modular decomposition: Linear time O(n+m) for finding modules

Key Functions:
- balanced_rank1_partition(G): Main algorithm for balanced rank-1 partitioning
- cut_rank(G, V_i): Compute cut-rank of a vertex set over GF(2)
- find_rank1_components(G): Find all rank-1 components in a graph
- verify_partition(G, partition): Verify partition properties
"""

import numpy as np
import networkx as nx
from typing import List, Set, Tuple, Optional, Dict, Union
from collections import defaultdict
from itertools import combinations
import warnings


# Type alias for vertex type (can be int or any hashable)
Vertex = Union[int, str]


def adjacency_matrix_gf2(G: nx.Graph) -> np.ndarray:
    """
    Compute the adjacency matrix of a graph over GF(2).
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    np.ndarray
        Adjacency matrix with entries in {0, 1}
    """
    return nx.to_numpy_array(G, dtype=np.int8) % 2


def gf2_rank(matrix: np.ndarray) -> int:
    """
    Compute the rank of a matrix over GF(2) using Gaussian elimination.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix with entries in {0, 1}
        
    Returns
    -------
    int
        Rank of the matrix over GF(2)
    """
    if matrix.size == 0:
        return 0
    
    # Work with a copy to avoid modifying original
    M = matrix.copy().astype(np.int8) % 2
    rows, cols = M.shape
    
    rank = 0
    pivot_col = 0
    
    for row in range(rows):
        if pivot_col >= cols:
            break
            
        # Find pivot in current column
        pivot_row = None
        for r in range(row, rows):
            if M[r, pivot_col] == 1:
                pivot_row = r
                break
        
        if pivot_row is None:
            pivot_col += 1
            continue
            
        # Swap rows if needed
        if pivot_row != row:
            M[[row, pivot_row]] = M[[pivot_row, row]]
        
        # Eliminate other rows
        for r in range(rows):
            if r != row and M[r, pivot_col] == 1:
                M[r] = (M[r] + M[row]) % 2
        
        rank += 1
        pivot_col += 1
    
    return rank


def cut_rank(G: nx.Graph, V_i: Set[int], nodes_list: List[int] = None) -> int:
    """
    Compute the cut-rank of a vertex set V_i relative to V \\ V_i over GF(2).
    
    The cut-rank is the rank of the submatrix A[V_i, V \\ V_i] where A is the
    adjacency matrix over GF(2).
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    V_i : Set[int]
        Subset of vertices
    nodes_list : List[int], optional
        Ordered list of nodes (if None, uses list(G.nodes()))
        
    Returns
    -------
    int
        Cut-rank of the partition (V_i, V \ V_i)
    """
    if nodes_list is None:
        nodes_list = list(G.nodes())
    
    V_complement = set(nodes_list) - V_i
    
    if not V_i or not V_complement:
        return 0
    
    # Get indices
    V_i_list = list(V_i)
    V_c_list = list(V_complement)
    
    # Build cut matrix A[V_i, V \ V_i]
    cut_matrix = np.zeros((len(V_i_list), len(V_c_list)), dtype=np.int8)
    
    for i, u in enumerate(V_i_list):
        for j, v in enumerate(V_c_list):
            if G.has_edge(u, v):
                cut_matrix[i, j] = 1
    
    return gf2_rank(cut_matrix)


def is_rank1_cut(G: nx.Graph, V_i: Set[int], nodes_list: List[int] = None) -> bool:
    """
    Check if a vertex set V_i forms a rank-1 cut.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    V_i : Set[int]
        Subset of vertices
    nodes_list : List[int], optional
        Ordered list of nodes
        
    Returns
    -------
    bool
        True if cut-rank equals 1
    """
    return cut_rank(G, V_i, nodes_list) == 1


def find_modules(G: nx.Graph) -> List[Set[int]]:
    """
    Find all non-trivial modules (twin sets) in the graph.
    
    Twin sets are nodes that share the exact same neighbors, excluding themselves
    
    A module M is a set of vertices such that every vertex outside M
    is either adjacent to all vertices in M or to none of them.
    Vertices in a module form a rank-1 cut (they have identical external neighborhoods).
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    List[Set[int]]
        List of modules (twin sets)
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return []
    
    # Build neighborhood signatures
    # Two vertices are twins if they have the same neighborhood (excluding each other)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Compute neighborhood for each node
    neighborhoods = {}
    for node in nodes:
        neighborhoods[node] = frozenset(G.neighbors(node))
    
    # Group vertices by their "external neighborhood signature"
    # For true twins: same closed neighborhood
    # For false twins: same open neighborhood (excluding each other)
    
    # Find true twins (same closed neighborhood)
    true_twin_groups = defaultdict(set)
    for node in nodes:
        # Closed neighborhood includes the node itself conceptually for comparison
        signature = frozenset(neighborhoods[node])
        true_twin_groups[signature].add(node)
    
    # Find false twins (same open neighborhood, not adjacent to each other)
    false_twin_groups = defaultdict(set)
    for node in nodes:
        signature = frozenset(neighborhoods[node])
        false_twin_groups[signature].add(node)
    
    # Collect non-trivial modules (size >= 2)
    modules = []
    seen = set()
    
    for signature, group in true_twin_groups.items():
        if len(group) >= 2:
            fg = frozenset(group)
            if fg not in seen:
                modules.append(group)
                seen.add(fg)
    
    return modules


def find_rank1_components(G: nx.Graph) -> List[Set[int]]:
    """
    Find vertex sets that form rank-1 cuts with the rest of the graph.
    
    This includes:
    1. Twin sets (modules) - vertices with identical neighborhoods
    2. Single vertices connected to the rest via a single "interface" vertex
    3. Pendant vertices (neighborhood = 1) and their neighbors
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    List[Set[int]]
        List of vertex sets, each forming a rank-1 cut
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n <= 1:
        return [set(nodes)] if nodes else []
    
    rank1_sets = []
    
    # 1. Find modules (twin sets)
    modules = find_modules(G)
    for module in modules:
        if is_rank1_cut(G, module, nodes):
            rank1_sets.append(module)
    
    # 2. Find single vertices that form rank-1 cuts
    # A single vertex {v} has cut-rank 1 if it has at least one neighbor
    for node in nodes:
        if G.degree(node) > 0:
            single_set = {node}
            # Single vertex always has cut-rank at most 1 (it's either 0 or 1)
            if cut_rank(G, single_set, nodes) == 1:
                rank1_sets.append(single_set)
    
    # 3. Find pendant structures (leaf + neighbor can sometimes form rank-1)
    leaves = [v for v in nodes if G.degree(v) == 1]
    for leaf in leaves:
        neighbor = list(G.neighbors(leaf))[0]
        pendant_set = {leaf, neighbor}
        if is_rank1_cut(G, pendant_set, nodes):
            rank1_sets.append(pendant_set)
    
    # 4. Exhaustive search for small rank-1 cuts (for small graphs)
    if n <= 20:
        for size in range(2, n // 2 + 1):
            for subset in combinations(nodes, size):
                subset_set = set(subset)
                if subset_set not in rank1_sets and is_rank1_cut(G, subset_set, nodes):
                    rank1_sets.append(subset_set)
    
    # Remove duplicates
    unique_sets = []
    seen = set()
    for s in rank1_sets:
        fs = frozenset(s)
        if fs not in seen:
            unique_sets.append(s)
            seen.add(fs)
    
    return unique_sets


def partition_variance(partition: List[Set[int]]) -> float:
    """
    Compute the variance of partition sizes.
    
    Parameters
    ----------
    partition : List[Set[int]]
        A partition of vertices
        
    Returns
    -------
    float
        Variance of the sizes
    """
    if not partition:
        return 0.0
    
    sizes = [len(p) for p in partition]
    return np.var(sizes)


def greedy_balanced_partition(
    G: nx.Graph,
    rank1_sets: List[Set[int]],
    target_num_parts: int = None
) -> Tuple[List[Set[int]], float]:
    """
    Greedy algorithm to create a balanced partition using rank-1 components.
    
    Uses a bin-packing inspired approach: iteratively assign vertices to bins
    trying to minimize variance.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    rank1_sets : List[Set[int]]
        Available rank-1 components
    target_num_parts : int, optional
        Target number of partitions (if None, automatically determined)
        
    Returns
    -------
    Tuple[List[Set[int]], float]
        (partition, variance)
    """
    nodes = set(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return [], 0.0
    
    if target_num_parts is None:
        # Heuristic: aim for sqrt(n) partitions
        target_num_parts = max(2, int(np.sqrt(n)))
    
    target_size = n / target_num_parts
    
    # Sort rank-1 sets by size (largest first for better packing)
    sorted_sets = sorted(rank1_sets, key=len, reverse=True)
    
    # Initialize bins
    bins = [set() for _ in range(target_num_parts)]
    used_vertices = set()
    
    # Greedy assignment: assign each rank-1 set to the bin with smallest size
    for r1_set in sorted_sets:
        # Skip if any vertex already assigned
        if r1_set & used_vertices:
            continue
        
        # Find bin with minimum size
        min_idx = min(range(len(bins)), key=lambda i: len(bins[i]))
        
        # Only add if it improves or maintains balance
        bins[min_idx].update(r1_set)
        used_vertices.update(r1_set)
    
    # Assign remaining vertices (those not in any rank-1 set)
    remaining = nodes - used_vertices
    for v in remaining:
        min_idx = min(range(len(bins)), key=lambda i: len(bins[i]))
        bins[min_idx].add(v)
    
    # Remove empty bins
    partition = [b for b in bins if len(b) > 0]
    
    # Verify all vertices are assigned
    assert set.union(*partition) == nodes, "Partition incomplete"
    
    return partition, partition_variance(partition)


def local_search_refinement(
    G: nx.Graph,
    partition: List[Set[int]],
    max_iterations: int = 1000
) -> Tuple[List[Set[int]], float]:
    """
    Local search to improve partition balance while maintaining rank-1 property.
    
    Attempts to move vertices between partitions to reduce variance.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set[int]]
        Initial partition
    max_iterations : int
        Maximum number of iterations
        
    Returns
    -------
    Tuple[List[Set[int]], float]
        (improved partition, variance)
    """
    if len(partition) <= 1:
        return partition, partition_variance(partition)
    
    nodes_list = list(G.nodes())
    current_partition = [p.copy() for p in partition]
    current_var = partition_variance(current_partition)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try moving each vertex to a different partition
        for part_idx, part in enumerate(current_partition):
            if len(part) <= 1:
                continue
                
            for v in list(part):
                # Find best target partition for v
                best_move = None
                best_var = current_var
                
                for target_idx in range(len(current_partition)):
                    if target_idx == part_idx:
                        continue
                    
                    # Simulate move
                    new_partition = [p.copy() for p in current_partition]
                    new_partition[part_idx].remove(v)
                    new_partition[target_idx].add(v)
                    
                    # Check if both affected parts still have rank <= 1
                    # (relaxed condition - we prioritize balance)
                    new_var = partition_variance(new_partition)
                    
                    if new_var < best_var - 1e-10:
                        best_var = new_var
                        best_move = target_idx
                
                if best_move is not None:
                    current_partition[part_idx].remove(v)
                    current_partition[best_move].add(v)
                    current_var = best_var
                    improved = True
    
    # Remove empty partitions
    current_partition = [p for p in current_partition if len(p) > 0]
    
    return current_partition, partition_variance(current_partition)


def simulated_annealing_partition(
    G: nx.Graph,
    initial_partition: List[Set[int]],
    temperature: float = 1000.0,
    cooling_rate: float = 0.995,
    max_iterations: int = 10000
) -> Tuple[List[Set[int]], float]:
    """
    Simulated annealing to find a balanced partition.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    initial_partition : List[Set[int]]
        Starting partition
    temperature : float
        Initial temperature
    cooling_rate : float
        Temperature decay rate
    max_iterations : int
        Maximum iterations
        
    Returns
    -------
    Tuple[List[Set[int]], float]
        (best partition, variance)
    """
    if len(initial_partition) <= 1:
        return initial_partition, partition_variance(initial_partition)
    
    current = [p.copy() for p in initial_partition]
    current_var = partition_variance(current)
    
    best = [p.copy() for p in current]
    best_var = current_var
    
    for iteration in range(max_iterations):
        # Generate neighbor: move a random vertex to a random partition
        non_empty = [i for i, p in enumerate(current) if len(p) > 1]
        if not non_empty:
            break
            
        # Pick source partition (must have at least 2 vertices)
        src_idx = np.random.choice(non_empty)
        src_vertex = np.random.choice(list(current[src_idx]))
        
        # Pick target partition (different from source)
        candidates = [i for i in range(len(current)) if i != src_idx]
        if not candidates:
            continue
        tgt_idx = np.random.choice(candidates)
        
        # Make move
        new_partition = [p.copy() for p in current]
        new_partition[src_idx].remove(src_vertex)
        new_partition[tgt_idx].add(src_vertex)
        new_var = partition_variance(new_partition)
        
        # Accept or reject
        delta = new_var - current_var
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            current = new_partition
            current_var = new_var
            
            if current_var < best_var:
                best = [p.copy() for p in current]
                best_var = current_var
        
        temperature *= cooling_rate
    
    # Remove empty partitions
    best = [p for p in best if len(p) > 0]
    
    return best, partition_variance(best)


def joint_objective(
    G: nx.Graph,
    partition: List[Set[int]],
    alpha: float = 0.5,
    nodes_list: List[int] = None
) -> float:
    """
    Joint objective function balancing rank-1 constraint and size balance.
    
    Objective = alpha * (sum of excess cut-ranks) + (1-alpha) * size_variance

    alpha is basically the chosen balance between the two objectives.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set[int]]
        Current partition
    alpha : float
        Weight for rank constraint (0 = balance only, 1 = rank only)
    nodes_list : List[int], optional
        Ordered list of nodes
        
    Returns
    -------
    float
        Combined objective value (lower is better)
    """
    if nodes_list is None:
        nodes_list = list(G.nodes())
    
    # Rank penalty: sum of (cut_rank - 1) for each part, clamped to 0
    rank_penalty = 0
    for part in partition:
        cr = cut_rank(G, part, nodes_list)
        rank_penalty += max(0, cr - 1)
    
    # Normalize by number of partitions
    if len(partition) > 0:
        rank_penalty /= len(partition)
    
    # Balance penalty: normalized variance
    sizes = [len(p) for p in partition]
    if len(sizes) > 0:
        mean_size = np.mean(sizes)
        if mean_size > 0:
            # Coefficient of variation squared
            balance_penalty = np.var(sizes) / (mean_size ** 2)
        else:
            balance_penalty = 0.0
    else:
        balance_penalty = 0.0
    
    return alpha * rank_penalty + (1 - alpha) * balance_penalty


def simulated_annealing_joint(
    G: nx.Graph,
    initial_partition: List[Set[int]],
    alpha: float = 0.5,
    temperature: float = 100.0,
    cooling_rate: float = 0.995,
    max_iterations: int = 20000
) -> Tuple[List[Set[int]], Dict]:
    """
    Simulated annealing optimizing the joint objective.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    initial_partition : List[Set[int]]
        Starting partition
    alpha : float
        Weight for rank-1 constraint vs balance (0.5 = equal weight)
    temperature : float
        Initial temperature
    cooling_rate : float
        Temperature decay rate
    max_iterations : int
        Maximum iterations
        
    Returns
    -------
    Tuple[List[Set[int]], Dict]
        (best partition, optimization info)
    """
    if len(initial_partition) <= 1:
        return initial_partition, {'final_obj': 0.0}
    
    nodes_list = list(G.nodes())
    
    current = [p.copy() for p in initial_partition]
    current_obj = joint_objective(G, current, alpha, nodes_list)
    
    best = [p.copy() for p in current]
    best_obj = current_obj
    
    history = []
    
    for iteration in range(max_iterations):
        # Generate neighbor
        non_empty = [i for i, p in enumerate(current) if len(p) > 1]
        if not non_empty:
            break
        
        src_idx = np.random.choice(non_empty)
        src_vertex = np.random.choice(list(current[src_idx]))
        
        candidates = [i for i in range(len(current)) if i != src_idx]
        if not candidates:
            continue
        tgt_idx = np.random.choice(candidates)
        
        # Make move
        new_partition = [p.copy() for p in current]
        new_partition[src_idx].remove(src_vertex)
        new_partition[tgt_idx].add(src_vertex)
        new_obj = joint_objective(G, new_partition, alpha, nodes_list)
        
        # Accept or reject
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / max(temperature, 1e-10)):
            current = new_partition
            current_obj = new_obj
            
            if current_obj < best_obj:
                best = [p.copy() for p in current]
                best_obj = current_obj
        
        temperature *= cooling_rate
        
        if iteration % 1000 == 0:
            history.append((iteration, best_obj, temperature))
    
    # Remove empty partitions
    best = [p for p in best if len(p) > 0]
    
    return best, {
        'final_objective': best_obj,
        'history': history,
        'alpha': alpha
    }


def rank1_preserving_local_search(
    G: nx.Graph,
    partition: List[Set[int]],
    max_iterations: int = 1000
) -> Tuple[List[Set[int]], float]:
    """
    Local search that tries to improve balance while preserving/improving rank-1 property.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set[int]]
        Initial partition
    max_iterations : int
        Maximum iterations
        
    Returns
    -------
    Tuple[List[Set[int]], float]
        (improved partition, variance)
    """
    if len(partition) <= 1:
        return partition, partition_variance(partition)
    
    nodes_list = list(G.nodes())
    current = [p.copy() for p in partition]
    
    # Compute initial cut-ranks
    current_ranks = [cut_rank(G, p, nodes_list) for p in current]
    current_var = partition_variance(current)
    
    for iteration in range(max_iterations):
        improved = False
        
        for part_idx, part in enumerate(current):
            if len(part) <= 1:
                continue
            
            for v in list(part):
                for target_idx in range(len(current)):
                    if target_idx == part_idx:
                        continue
                    
                    # Simulate move
                    new_partition = [p.copy() for p in current]
                    new_partition[part_idx].remove(v)
                    new_partition[target_idx].add(v)
                    
                    # Compute new ranks for affected partitions
                    new_rank_src = cut_rank(G, new_partition[part_idx], nodes_list)
                    new_rank_tgt = cut_rank(G, new_partition[target_idx], nodes_list)
                    
                    # Check if this improves or maintains rank-1 property
                    old_rank_violations = (current_ranks[part_idx] > 1) + (current_ranks[target_idx] > 1)
                    new_rank_violations = (new_rank_src > 1) + (new_rank_tgt > 1)
                    
                    new_var = partition_variance(new_partition)
                    
                    # Accept if: fewer rank violations, OR same violations but better balance
                    accept = False
                    if new_rank_violations < old_rank_violations:
                        accept = True
                    elif new_rank_violations == old_rank_violations and new_var < current_var - 1e-10:
                        accept = True
                    
                    if accept:
                        current = new_partition
                        current_ranks[part_idx] = new_rank_src
                        current_ranks[target_idx] = new_rank_tgt
                        current_var = new_var
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current, partition_variance(current)


#main algo:
def balanced_rank1_partition(
    G: nx.Graph,
    target_num_parts: int = None,
    use_local_search: bool = True,
    use_simulated_annealing: bool = True,
    use_joint_optimization: bool = False,
    alpha: float = 0.5,
    verbose: bool = True
) -> Dict:
    """
    Main algorithm for the Balanced Rank-1 Partitioning Problem.
    
    Finds a partition of the graph vertices such that:
    1. Each part ideally forms a rank-1 cut with the rest (or approximates this)
    2. The partition is balanced (minimizes size variance)
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    target_num_parts : int, optional
        Target number of partitions
    use_local_search : bool
        Whether to apply local search refinement
    use_simulated_annealing : bool
        Whether to apply simulated annealing
    use_joint_optimization : bool
        Whether to use joint objective optimization (balances rank-1 and size)
    alpha : float
        Weight for rank-1 constraint when using joint optimization (0-1)
        Higher alpha = more emphasis on rank-1 property
    verbose : bool
        Print progress information
        
    Returns
    -------
    Dict
        Dictionary containing:
        - 'partition': List[Set[int]] - the final partition
        - 'variance': float - variance of partition sizes
        - 'sizes': List[int] - sizes of each partition
        - 'rank1_sets_found': int - number of rank-1 components found
        - 'cut_ranks': List[int] - cut-rank of each partition
        - 'num_rank1_parts': int - number of partitions with cut-rank = 1
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    if verbose:
        print(f"Graph has {n} vertices and {G.number_of_edges()} edges")
    
    if n == 0:
        return {
            'partition': [],
            'variance': 0.0,
            'sizes': [],
            'rank1_sets_found': 0,
            'cut_ranks': [],
            'num_rank1_parts': 0
        }
    
    # Step 1: Find rank-1 components
    if verbose:
        print("Finding rank-1 components...")
    rank1_sets = find_rank1_components(G)
    if verbose:
        print(f"Found {len(rank1_sets)} rank-1 components")
    
    # Step 2: Greedy balanced partition
    if verbose:
        print("Computing greedy balanced partition...")
    partition, var = greedy_balanced_partition(G, rank1_sets, target_num_parts)
    if verbose:
        print(f"Initial partition: {len(partition)} parts, variance = {var:.4f}")
    
    # Step 3: Local search refinement (rank-1 preserving)
    if use_local_search:
        if verbose:
            print("Applying rank-1 preserving local search...")
        partition, var = rank1_preserving_local_search(G, partition)
        if verbose:
            print(f"After local search: variance = {var:.4f}")
    
    # Step 4: Optimization
    if use_joint_optimization:
        if verbose:
            print(f"Applying joint optimization (alpha={alpha})...")
        partition, opt_info = simulated_annealing_joint(G, partition, alpha=alpha)
        var = partition_variance(partition)
        if verbose:
            print(f"After joint optimization: variance = {var:.4f}")
    elif use_simulated_annealing:
        if verbose:
            print("Applying simulated annealing...")
        partition, var = simulated_annealing_partition(G, partition)
        if verbose:
            print(f"After simulated annealing: variance = {var:.4f}")
    
    # Compute cut-ranks for final partition
    cut_ranks = []
    for part in partition:
        cr = cut_rank(G, part, nodes)
        cut_ranks.append(cr)
    
    sizes = [len(p) for p in partition]
    num_rank1 = sum(1 for cr in cut_ranks if cr == 1)
    
    if verbose:
        print(f"\nFinal Result:")
        print(f"  Number of partitions: {len(partition)}")
        print(f"  Partition sizes: {sizes}")
        print(f"  Size variance: {var:.4f}")
        print(f"  Cut-ranks: {cut_ranks}")
        print(f"  Partitions with rank-1: {num_rank1}/{len(partition)}")
        print(f"  All rank-1: {all(cr == 1 for cr in cut_ranks)}")
    
    return {
        'partition': partition,
        'variance': var,
        'sizes': sizes,
        'rank1_sets_found': len(rank1_sets),
        'cut_ranks': cut_ranks,
        'num_rank1_parts': num_rank1
    }


def verify_partition(G: nx.Graph, partition: List[Set[int]]) -> Dict:
    """
    Verify properties of a partition.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set[int]]
        Partition to verify
        
    Returns
    -------
    Dict
        Verification results
    """
    nodes = set(G.nodes())
    
    # Check completeness
    partition_nodes = set.union(*partition) if partition else set()
    is_complete = partition_nodes == nodes
    
    # Check disjointness
    is_disjoint = sum(len(p) for p in partition) == len(partition_nodes)
    
    # Compute cut-ranks
    nodes_list = list(G.nodes())
    cut_ranks = [cut_rank(G, p, nodes_list) for p in partition]
    all_rank1 = all(cr == 1 for cr in cut_ranks)
    
    # Compute balance metrics
    sizes = [len(p) for p in partition]
    variance = np.var(sizes) if sizes else 0.0
    max_size = max(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    
    return {
        'is_complete': is_complete,
        'is_disjoint': is_disjoint,
        'is_valid_partition': is_complete and is_disjoint,
        'all_rank1': all_rank1,
        'cut_ranks': cut_ranks,
        'sizes': sizes,
        'variance': variance,
        'max_size': max_size,
        'min_size': min_size,
        'size_ratio': max_size / min_size if min_size > 0 else float('inf')
    }


# ============================================================================
# Balance Metrics - Quantifying "Close Enough"
# ============================================================================

def compute_balance_metrics(sizes: List[int]) -> Dict[str, float]:
    """
    Compute comprehensive balance metrics for partition sizes.
    
    These metrics quantify how "close enough" the partition sizes are.
    
    Parameters
    ----------
    sizes : List[int]
        List of partition sizes
        
    Returns
    -------
    Dict[str, float]
        Dictionary with balance metrics:
        - 'variance': Raw variance of sizes
        - 'std_dev': Standard deviation
        - 'cv': Coefficient of Variation (std/mean) - scale-independent measure
        - 'size_ratio': max/min ratio (1.0 = perfect, higher = worse)
        - 'imbalance': (max-min)/mean - normalized spread
        - 'balance_score': Combined score in [0,1] where 1 = perfectly balanced
    """
    if not sizes or len(sizes) == 0:
        return {
            'variance': 0.0, 'std_dev': 0.0, 'cv': 0.0,
            'size_ratio': 1.0, 'imbalance': 0.0, 'balance_score': 1.0
        }
    
    sizes = np.array(sizes, dtype=float)
    n_parts = len(sizes)
    
    mean_size = np.mean(sizes)
    variance = np.var(sizes)
    std_dev = np.std(sizes)
    max_size = np.max(sizes)
    min_size = np.min(sizes)
    
    # Coefficient of Variation (CV): scale-independent measure
    # CV = 0 means perfect balance, CV > 1 means high imbalance
    cv = std_dev / mean_size if mean_size > 0 else 0.0
    
    # Size ratio: max/min (1.0 = perfect, >2.0 typically considered imbalanced)
    size_ratio = max_size / min_size if min_size > 0 else float('inf')
    
    # Imbalance: normalized spread
    imbalance = (max_size - min_size) / mean_size if mean_size > 0 else 0.0
    
    # Combined balance score in [0, 1] where 1 = perfectly balanced
    # Uses exponential decay based on CV and size_ratio
    # Tuneable thresholds: cv_threshold=0.5, ratio_threshold=2.0
    cv_score = np.exp(-2 * cv)  # cv=0 -> 1.0, cv=0.5 -> 0.37
    ratio_score = np.exp(-0.5 * (size_ratio - 1))  # ratio=1 -> 1.0, ratio=2 -> 0.6
    balance_score = (cv_score + ratio_score) / 2
    
    return {
        'variance': float(variance),
        'std_dev': float(std_dev),
        'cv': float(cv),
        'size_ratio': float(size_ratio),
        'imbalance': float(imbalance),
        'balance_score': float(balance_score)
    }


def is_balanced_enough(
    sizes: List[int],
    max_cv: float = 0.5,
    max_ratio: float = 2.0
) -> bool:
    """
    Check if partition sizes are "close enough" to balanced.
    
    Parameters
    ----------
    sizes : List[int]
        Partition sizes
    max_cv : float
        Maximum acceptable coefficient of variation (default 0.5)
    max_ratio : float
        Maximum acceptable max/min ratio (default 2.0)
        
    Returns
    -------
    bool
        True if partition is considered balanced enough
    """
    metrics = compute_balance_metrics(sizes)
    return metrics['cv'] <= max_cv and metrics['size_ratio'] <= max_ratio


# ============================================================================
# Universal Graph Partitioning (Works for ANY connected simple graph)
# ============================================================================

def universal_objective(
    G: nx.Graph,
    partition: List[Set[int]],
    nodes_list: List[int],
    rank_weight: float = 0.6,
    balance_weight: float = 0.4
) -> float:
    """
    Universal objective function for any graph (DH or non-DH).
    
    Combines:
    1. Total cut-rank penalty (lower is better)
    2. Balance penalty based on CV and size ratio
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set[int]]
        Current partition
    nodes_list : List[int]
        Ordered list of all nodes
    rank_weight : float
        Weight for rank minimization (default 0.6)
    balance_weight : float
        Weight for balance (default 0.4)
        
    Returns
    -------
    float
        Objective value (lower is better)
    """
    if not partition:
        return float('inf')
    
    # Compute cut-ranks
    total_rank = 0
    for part in partition:
        if len(part) > 0:
            cr = cut_rank(G, part, nodes_list)
            total_rank += cr
    
    # Normalize rank by number of partitions (ideal = 1 per partition)
    avg_rank = total_rank / len(partition)
    rank_penalty = max(0, avg_rank - 1)  # Penalty for exceeding rank-1
    
    # Balance metrics
    sizes = [len(p) for p in partition]
    metrics = compute_balance_metrics(sizes)
    
    # Balance penalty: combination of CV and size ratio deviation
    balance_penalty = metrics['cv'] + 0.5 * max(0, metrics['size_ratio'] - 1)
    
    return rank_weight * rank_penalty + balance_weight * balance_penalty


def greedy_mincut_partition(
    G: nx.Graph,
    target_num_parts: int
) -> List[Set[int]]:
    """
    Greedy partitioning that minimizes cut-rank at each step.
    
    Works by iteratively splitting the largest partition using
    the cut with minimum rank.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    target_num_parts : int
        Target number of partitions
        
    Returns
    -------
    List[Set[int]]
        Partition of vertices
    """
    nodes = list(G.nodes())
    n = len(nodes)
    nodes_set = set(nodes)
    
    if n == 0 or target_num_parts <= 0:
        return []
    
    if target_num_parts == 1:
        return [nodes_set]
    
    # Start with all nodes in one partition
    partition = [nodes_set.copy()]
    
    while len(partition) < target_num_parts:
        # Find the largest partition to split
        largest_idx = max(range(len(partition)), key=lambda i: len(partition[i]))
        to_split = partition[largest_idx]
        
        if len(to_split) <= 1:
            break  # Cannot split further
        
        # Find best split (minimize max cut-rank of resulting parts)
        best_split = None
        best_score = float('inf')
        
        to_split_list = list(to_split)
        
        # Try different split strategies
        # Strategy 1: BFS-based balanced split
        subgraph = G.subgraph(to_split)
        if nx.is_connected(subgraph):
            # Use spectral bisection approximation
            start_node = to_split_list[0]
            bfs_order = list(nx.bfs_tree(subgraph, start_node).nodes())
            mid = len(bfs_order) // 2
            part_a = set(bfs_order[:mid])
            part_b = set(bfs_order[mid:])
            
            if part_a and part_b:
                # Compute score (sum of cut-ranks + balance penalty)
                remaining = nodes_set - to_split
                temp_partition = [p for i, p in enumerate(partition) if i != largest_idx]
                temp_partition.extend([part_a, part_b])
                
                rank_a = cut_rank(G, part_a, nodes)
                rank_b = cut_rank(G, part_b, nodes)
                size_diff = abs(len(part_a) - len(part_b))
                score = rank_a + rank_b + 0.1 * size_diff
                
                if score < best_score:
                    best_score = score
                    best_split = (part_a, part_b)
        
        # Strategy 2: Try splitting by removing one vertex at a time
        # (for small partitions)
        if len(to_split) <= 10:
            for v in to_split_list:
                part_a = {v}
                part_b = to_split - {v}
                
                rank_a = cut_rank(G, part_a, nodes)
                rank_b = cut_rank(G, part_b, nodes)
                size_diff = abs(len(part_a) - len(part_b))
                score = rank_a + rank_b + 0.5 * size_diff
                
                if score < best_score:
                    best_score = score
                    best_split = (part_a, part_b)
        
        # Strategy 3: Random balanced splits
        for _ in range(min(20, len(to_split))):
            shuffled = to_split_list.copy()
            np.random.shuffle(shuffled)
            mid = len(shuffled) // 2
            part_a = set(shuffled[:mid])
            part_b = set(shuffled[mid:])
            
            if part_a and part_b:
                rank_a = cut_rank(G, part_a, nodes)
                rank_b = cut_rank(G, part_b, nodes)
                size_diff = abs(len(part_a) - len(part_b))
                score = rank_a + rank_b + 0.1 * size_diff
                
                if score < best_score:
                    best_score = score
                    best_split = (part_a, part_b)
        
        if best_split is None:
            # Fallback: arbitrary split
            mid = len(to_split_list) // 2
            best_split = (set(to_split_list[:mid]), set(to_split_list[mid:]))
        
        # Apply the split
        partition.pop(largest_idx)
        partition.append(best_split[0])
        partition.append(best_split[1])
    
    return partition


def universal_simulated_annealing(
    G: nx.Graph,
    initial_partition: List[Set[int]],
    rank_weight: float = 0.6,
    balance_weight: float = 0.4,
    temperature: float = 50.0,
    cooling_rate: float = 0.997,
    max_iterations: int = 30000
) -> Tuple[List[Set[int]], Dict]:
    """
    Simulated annealing for universal graph partitioning.
    
    Optimizes the universal objective that balances cut-rank minimization
    with partition balance.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    initial_partition : List[Set[int]]
        Starting partition
    rank_weight : float
        Weight for rank minimization
    balance_weight : float
        Weight for balance
    temperature : float
        Initial temperature
    cooling_rate : float
        Temperature decay rate
    max_iterations : int
        Maximum iterations
        
    Returns
    -------
    Tuple[List[Set[int]], Dict]
        (best partition, optimization info)
    """
    if len(initial_partition) <= 1:
        return initial_partition, {'iterations': 0}
    
    nodes_list = list(G.nodes())
    
    current = [p.copy() for p in initial_partition]
    current_obj = universal_objective(G, current, nodes_list, rank_weight, balance_weight)
    
    best = [p.copy() for p in current]
    best_obj = current_obj
    
    no_improve_count = 0
    max_no_improve = 3000
    
    for iteration in range(max_iterations):
        # Generate neighbor move
        move_type = np.random.choice(['single', 'swap'], p=[0.7, 0.3])
        
        if move_type == 'single':
            # Move single vertex between partitions
            non_empty = [i for i, p in enumerate(current) if len(p) > 1]
            if not non_empty:
                continue
            
            src_idx = np.random.choice(non_empty)
            src_vertex = np.random.choice(list(current[src_idx]))
            
            candidates = [i for i in range(len(current)) if i != src_idx]
            if not candidates:
                continue
            tgt_idx = np.random.choice(candidates)
            
            new_partition = [p.copy() for p in current]
            new_partition[src_idx].remove(src_vertex)
            new_partition[tgt_idx].add(src_vertex)
            
        else:  # swap
            # Swap vertices between two partitions
            non_empty = [i for i, p in enumerate(current) if len(p) >= 1]
            if len(non_empty) < 2:
                continue
            
            idx1, idx2 = np.random.choice(non_empty, size=2, replace=False)
            v1 = np.random.choice(list(current[idx1]))
            v2 = np.random.choice(list(current[idx2]))
            
            new_partition = [p.copy() for p in current]
            new_partition[idx1].remove(v1)
            new_partition[idx1].add(v2)
            new_partition[idx2].remove(v2)
            new_partition[idx2].add(v1)
        
        # Remove empty partitions
        new_partition = [p for p in new_partition if len(p) > 0]
        if len(new_partition) == 0:
            continue
        
        new_obj = universal_objective(G, new_partition, nodes_list, rank_weight, balance_weight)
        
        # Accept or reject
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / max(temperature, 1e-10)):
            current = new_partition
            current_obj = new_obj
            
            if current_obj < best_obj:
                best = [p.copy() for p in current]
                best_obj = current_obj
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= max_no_improve:
            break
        
        temperature *= cooling_rate
    
    return best, {
        'final_objective': best_obj,
        'iterations': iteration + 1,
        'rank_weight': rank_weight,
        'balance_weight': balance_weight
    }


def universal_partition(
    G: nx.Graph,
    target_num_parts: int = None,
    rank_weight: float = 0.6,
    balance_weight: float = 0.4,
    verbose: bool = True
) -> Dict:
    """
    Universal partitioning algorithm for ANY connected simple graph.
    
    This algorithm works for both distance-hereditary and non-DH graphs:
    - For DH graphs: Will naturally find rank-1 partitions
    - For non-DH graphs: Minimizes cut-rank while maintaining balance
    
    The algorithm prioritizes:
    1. Low cut-rank (ideally rank-1, but accepts higher when necessary)
    2. Balanced partition sizes ("close enough" as defined by metrics)
    
    Parameters
    ----------
    G : nx.Graph
        Input graph (must be connected)
    target_num_parts : int, optional
        Target number of partitions (default: sqrt(n))
    rank_weight : float
        Weight for rank minimization in objective (default 0.6)
    balance_weight : float
        Weight for balance in objective (default 0.4)
    verbose : bool
        Print progress information
        
    Returns
    -------
    Dict
        Results including partition, metrics, and balance information
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return {
            'partition': [],
            'cut_ranks': [],
            'sizes': [],
            'balance_metrics': compute_balance_metrics([]),
            'total_cut_rank': 0,
            'avg_cut_rank': 0.0,
            'is_distance_hereditary': True,
            'algorithm_used': 'none'
        }
    
    # Check connectivity
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected")
    
    # Determine target number of partitions
    if target_num_parts is None:
        target_num_parts = max(2, int(np.sqrt(n)))
    
    target_num_parts = min(target_num_parts, n)
    
    if verbose:
        print(f"Graph: {n} vertices, {G.number_of_edges()} edges")
        print(f"Target partitions: {target_num_parts}")
    
    # Check if distance-hereditary
    is_dh = is_distance_hereditary(G)
    if verbose:
        print(f"Distance-hereditary: {is_dh}")
    
    if is_dh:
        # Use optimized DH algorithm
        if verbose:
            print("Using rank-1 optimized algorithm for DH graph...")
        
        rank1_sets = find_rank1_components(G)
        partition, _ = greedy_balanced_partition(G, rank1_sets, target_num_parts)
        partition, _ = rank1_preserving_local_search(G, partition)
        partition, _ = simulated_annealing_joint(G, partition, alpha=0.7)
        algorithm_used = 'dh_rank1_optimized'
    else:
        # Use universal algorithm for non-DH graphs
        if verbose:
            print("Using universal algorithm for non-DH graph...")
        
        # Step 1: Greedy initial partition minimizing cut-rank
        partition = greedy_mincut_partition(G, target_num_parts)
        
        if verbose:
            initial_ranks = [cut_rank(G, p, nodes) for p in partition]
            print(f"Initial cut-ranks: {initial_ranks}")
        
        # Step 2: Optimize with SA
        partition, opt_info = universal_simulated_annealing(
            G, partition,
            rank_weight=rank_weight,
            balance_weight=balance_weight,
            max_iterations=30000
        )
        
        algorithm_used = 'universal_sa'
    
    # Compute final metrics
    cut_ranks = [cut_rank(G, p, nodes) for p in partition]
    sizes = [len(p) for p in partition]
    balance_metrics = compute_balance_metrics(sizes)
    total_rank = sum(cut_ranks)
    avg_rank = total_rank / len(partition) if partition else 0
    
    num_rank1 = sum(1 for cr in cut_ranks if cr == 1)
    
    if verbose:
        print(f"\n{'='*50}")
        print("FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Number of partitions: {len(partition)}")
        print(f"Partition sizes: {sizes}")
        print(f"Cut-ranks: {cut_ranks}")
        print(f"Total cut-rank: {total_rank}")
        print(f"Average cut-rank: {avg_rank:.2f}")
        print(f"Rank-1 partitions: {num_rank1}/{len(partition)}")
        print(f"\nBalance Metrics:")
        print(f"  Variance: {balance_metrics['variance']:.4f}")
        print(f"  Std Dev: {balance_metrics['std_dev']:.4f}")
        print(f"  CV (Coef. of Variation): {balance_metrics['cv']:.4f}")
        print(f"  Size Ratio (max/min): {balance_metrics['size_ratio']:.4f}")
        print(f"  Imbalance: {balance_metrics['imbalance']:.4f}")
        print(f"  Balance Score: {balance_metrics['balance_score']:.4f}")
        print(f"  Is 'Close Enough': {is_balanced_enough(sizes)}")
    
    return {
        'partition': partition,
        'cut_ranks': cut_ranks,
        'sizes': sizes,
        'total_cut_rank': total_rank,
        'avg_cut_rank': avg_rank,
        'num_rank1_parts': num_rank1,
        'balance_metrics': balance_metrics,
        'is_distance_hereditary': is_dh,
        'algorithm_used': algorithm_used,
        'is_balanced_enough': is_balanced_enough(sizes)
    }


# ============================================================================
# Minimum Linear Arrangement (MinLA) Support Functions
# ============================================================================

def compute_inter_partition_edges(
    G: nx.Graph,
    partition: List[Set]
) -> Dict[Tuple[int, int], List[Tuple]]:
    """
    Compute all edges between different partitions.
    
    These are the "interface" edges that will cross partition boundaries
    in the linear arrangement.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set]
        Partition of vertices
        
    Returns
    -------
    Dict[Tuple[int, int], List[Tuple]]
        Dictionary mapping (part_i, part_j) to list of edges between them
    """
    # Build vertex -> partition index mapping
    vertex_to_part = {}
    for i, part in enumerate(partition):
        for v in part:
            vertex_to_part[v] = i
    
    inter_edges = defaultdict(list)
    
    for u, v in G.edges():
        part_u = vertex_to_part.get(u)
        part_v = vertex_to_part.get(v)
        
        if part_u is not None and part_v is not None and part_u != part_v:
            # Normalize edge key (smaller index first)
            key = (min(part_u, part_v), max(part_u, part_v))
            inter_edges[key].append((u, v))
    
    return dict(inter_edges)


def build_partition_graph(
    G: nx.Graph,
    partition: List[Set]
) -> nx.Graph:
    """
    Build a weighted graph where nodes are partitions and edge weights
    are the number of edges between partitions.
    
    This "quotient graph" captures the connectivity structure between partitions.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set]
        Partition of vertices
        
    Returns
    -------
    nx.Graph
        Partition graph with edge weights
    """
    inter_edges = compute_inter_partition_edges(G, partition)
    
    P = nx.Graph()
    P.add_nodes_from(range(len(partition)))
    
    # Add node attributes (partition size, vertices)
    for i, part in enumerate(partition):
        P.nodes[i]['size'] = len(part)
        P.nodes[i]['vertices'] = part
    
    # Add weighted edges
    for (i, j), edges in inter_edges.items():
        P.add_edge(i, j, weight=len(edges), edges=edges)
    
    return P


def compute_partition_interface(
    G: nx.Graph,
    partition: List[Set]
) -> Dict[int, Dict]:
    """
    For each partition, compute its interface vertices and their connections.
    
    Interface vertices are those with edges to other partitions.
    This information is crucial for MinLA as these vertices should be
    positioned near partition boundaries.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set]
        Partition of vertices
        
    Returns
    -------
    Dict[int, Dict]
        For each partition index, returns:
        - 'interface_vertices': set of vertices with external edges
        - 'internal_vertices': set of vertices with only internal edges
        - 'connections': dict mapping interface vertex to list of (neighbor, part_idx)
    """
    vertex_to_part = {}
    for i, part in enumerate(partition):
        for v in part:
            vertex_to_part[v] = i
    
    result = {}
    
    for i, part in enumerate(partition):
        interface_vertices = set()
        internal_vertices = set()
        connections = defaultdict(list)
        
        for v in part:
            has_external = False
            for neighbor in G.neighbors(v):
                neighbor_part = vertex_to_part.get(neighbor)
                if neighbor_part is not None and neighbor_part != i:
                    has_external = True
                    connections[v].append((neighbor, neighbor_part))
            
            if has_external:
                interface_vertices.add(v)
            else:
                internal_vertices.add(v)
        
        result[i] = {
            'interface_vertices': interface_vertices,
            'internal_vertices': internal_vertices,
            'connections': dict(connections)
        }
    
    return result


def optimal_partition_ordering_greedy(
    G: nx.Graph,
    partition: List[Set]
) -> List[int]:
    """
    Find a good ordering of partitions for MinLA using a greedy approach.
    
    The goal is to order partitions such that partitions with many edges
    between them are placed adjacent in the ordering.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set]
        Partition of vertices
        
    Returns
    -------
    List[int]
        Ordering of partition indices
    """
    n_parts = len(partition)
    if n_parts <= 1:
        return list(range(n_parts))
    
    P = build_partition_graph(G, partition)
    
    # Greedy: start from partition with most edges, always pick adjacent with most shared edges
    # This is similar to nearest-neighbor heuristic for TSP
    
    remaining = set(range(n_parts))
    
    # Start with the partition that has most total inter-partition edges
    def total_weight(i):
        return sum(P[i][j]['weight'] for j in P.neighbors(i))
    
    start = max(remaining, key=total_weight)
    ordering = [start]
    remaining.remove(start)
    
    while remaining:
        last = ordering[-1]
        
        # Find remaining partition with most edges to current last partition
        best_next = None
        best_weight = -1
        
        for candidate in remaining:
            if P.has_edge(last, candidate):
                weight = P[last][candidate]['weight']
            else:
                weight = 0
            
            if weight > best_weight:
                best_weight = weight
                best_next = candidate
        
        if best_next is None:
            # No connected partition, pick any
            best_next = next(iter(remaining))
        
        ordering.append(best_next)
        remaining.remove(best_next)
    
    return ordering


def optimal_partition_ordering_tsp(
    G: nx.Graph,
    partition: List[Set]
) -> List[int]:
    """
    Find optimal partition ordering using TSP-like approach.
    
    Solves approximately the problem of ordering partitions to minimize
    the sum of distances between connected partitions.
    
    For small number of partitions, tries all permutations.
    For larger, uses 2-opt local search.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set]
        Partition of vertices
        
    Returns
    -------
    List[int]
        Ordering of partition indices
    """
    n_parts = len(partition)
    if n_parts <= 1:
        return list(range(n_parts))
    
    P = build_partition_graph(G, partition)
    inter_edges = compute_inter_partition_edges(G, partition)
    
    def compute_cost(ordering: List[int]) -> float:
        """
        Compute MinLA-like cost: sum of (distance * edge_count) for all inter-partition edges.
        """
        pos = {part_idx: i for i, part_idx in enumerate(ordering)}
        cost = 0
        for (i, j), edges in inter_edges.items():
            distance = abs(pos[i] - pos[j])
            cost += distance * len(edges)
        return cost
    
    # For small n, try all permutations
    if n_parts <= 8:
        from itertools import permutations
        best_ordering = None
        best_cost = float('inf')
        
        for perm in permutations(range(n_parts)):
            cost = compute_cost(list(perm))
            if cost < best_cost:
                best_cost = cost
                best_ordering = list(perm)
        
        return best_ordering
    
    # For larger n, use greedy + 2-opt
    current = optimal_partition_ordering_greedy(G, partition)
    current_cost = compute_cost(current)
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n_parts - 1):
            for j in range(i + 2, n_parts):
                # Reverse segment [i+1, j]
                new_ordering = current[:i+1] + current[i+1:j+1][::-1] + current[j+1:]
                new_cost = compute_cost(new_ordering)
                
                if new_cost < current_cost:
                    current = new_ordering
                    current_cost = new_cost
                    improved = True
    
    return current


def order_vertices_within_partition(
    G: nx.Graph,
    part: Set,
    interface_info: Dict,
    left_neighbor_part: Optional[int] = None,
    right_neighbor_part: Optional[int] = None
) -> List:
    """
    Order vertices within a partition for MinLA.
    
    Strategy: Place interface vertices at the boundary sides where they connect.
    Internal vertices go in the middle.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    part : Set
        Vertices in this partition
    interface_info : Dict
        Interface information for this partition
    left_neighbor_part : int, optional
        Index of partition to the left in final ordering
    right_neighbor_part : int, optional
        Index of partition to the right in final ordering
        
    Returns
    -------
    List
        Ordered list of vertices
    """
    if len(part) <= 1:
        return list(part)
    
    interface_vertices = interface_info['interface_vertices']
    internal_vertices = interface_info['internal_vertices']
    connections = interface_info['connections']
    
    # Categorize interface vertices by which neighbor partition they connect to
    left_interface = set()
    right_interface = set()
    both_interface = set()
    
    for v in interface_vertices:
        connects_left = any(p == left_neighbor_part for _, p in connections.get(v, []))
        connects_right = any(p == right_neighbor_part for _, p in connections.get(v, []))
        
        if connects_left and connects_right:
            both_interface.add(v)
        elif connects_left:
            left_interface.add(v)
        elif connects_right:
            right_interface.add(v)
        else:
            # Connects to some other partition (not immediate neighbors)
            # Place in middle
            internal_vertices.add(v)
    
    # Build subgraph for ordering within partition
    subgraph = G.subgraph(part)
    
    # Order each group using BFS from a central vertex
    def bfs_order(vertices: Set) -> List:
        if not vertices:
            return []
        vertices = set(vertices)  # Copy
        sub = subgraph.subgraph(vertices)
        if not nx.is_connected(sub):
            # Handle disconnected components
            result = []
            for comp in nx.connected_components(sub):
                result.extend(bfs_order(comp))
            return result
        
        # Start BFS from vertex with highest degree in subgraph
        start = max(vertices, key=lambda v: sub.degree(v))
        return list(nx.bfs_tree(sub, start).nodes())
    
    # Combine: left_interface + both_interface + internal + right_interface
    ordered = []
    ordered.extend(bfs_order(left_interface))
    ordered.extend(bfs_order(both_interface))
    ordered.extend(bfs_order(internal_vertices))
    ordered.extend(bfs_order(right_interface))
    
    return ordered


def generate_linear_arrangement(
    G: nx.Graph,
    partition: List[Set],
    partition_ordering: List[int] = None
) -> Tuple[List, Dict]:
    """
    Generate a linear arrangement (vertex labeling) from the decomposition.
    
    This is the main function for using decomposition to solve MinLA.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    partition : List[Set]
        Partition of vertices
    partition_ordering : List[int], optional
        Order of partitions (if None, computed optimally)
        
    Returns
    -------
    Tuple[List, Dict]
        - List of vertices in linear order
        - Dict with arrangement metrics
    """
    if not partition:
        return [], {'total_edge_length': 0}
    
    # Get optimal partition ordering if not provided
    if partition_ordering is None:
        partition_ordering = optimal_partition_ordering_tsp(G, partition)
    
    # Compute interface information
    interface_info = compute_partition_interface(G, partition)
    
    # Order vertices within each partition
    linear_order = []
    
    for idx, part_idx in enumerate(partition_ordering):
        part = partition[part_idx]
        
        # Determine left and right neighbor partitions
        left_neighbor = partition_ordering[idx - 1] if idx > 0 else None
        right_neighbor = partition_ordering[idx + 1] if idx < len(partition_ordering) - 1 else None
        
        # Order vertices within this partition
        part_order = order_vertices_within_partition(
            G, part, interface_info[part_idx],
            left_neighbor, right_neighbor
        )
        
        linear_order.extend(part_order)
    
    # Compute arrangement quality metrics
    vertex_pos = {v: i for i, v in enumerate(linear_order)}
    
    total_edge_length = 0
    max_edge_length = 0
    edge_lengths = []
    
    for u, v in G.edges():
        if u in vertex_pos and v in vertex_pos:
            length = abs(vertex_pos[u] - vertex_pos[v])
            total_edge_length += length
            max_edge_length = max(max_edge_length, length)
            edge_lengths.append(length)
    
    avg_edge_length = np.mean(edge_lengths) if edge_lengths else 0
    
    return linear_order, {
        'total_edge_length': total_edge_length,
        'max_edge_length': max_edge_length,
        'avg_edge_length': float(avg_edge_length),
        'num_edges': len(edge_lengths),
        'partition_ordering': partition_ordering
    }


def compute_minla_cost(G: nx.Graph, ordering: List) -> int:
    """
    Compute the total edge length (MinLA cost) for a given vertex ordering.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    ordering : List
        Linear ordering of vertices
        
    Returns
    -------
    int
        Total edge length (sum of |pos(u) - pos(v)| for all edges)
    """
    pos = {v: i for i, v in enumerate(ordering)}
    total = 0
    for u, v in G.edges():
        if u in pos and v in pos:
            total += abs(pos[u] - pos[v])
    return total


def decomposition_for_minla(
    G: nx.Graph,
    target_num_parts: int = None,
    verbose: bool = True
) -> Dict:
    """
    Complete pipeline: decompose graph and generate MinLA-optimized labeling.
    
    This is the main entry point for using graph decomposition to solve MinLA.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph (must be connected)
    target_num_parts : int, optional
        Number of partitions (default: sqrt(n))
    verbose : bool
        Print progress information
        
    Returns
    -------
    Dict
        Complete results including:
        - 'partition': The graph partition
        - 'linear_order': Vertex ordering for MinLA
        - 'minla_cost': Total edge length
        - 'partition_graph': Quotient graph of partitions
        - 'interface_info': Interface vertex information
        - All metrics from universal_partition
    """
    if verbose:
        print("=" * 60)
        print("DECOMPOSITION FOR MINIMUM LINEAR ARRANGEMENT")
        print("=" * 60)
    
    # Step 1: Decompose the graph
    if verbose:
        print("\n[Step 1] Decomposing graph...")
    
    decomp_result = universal_partition(
        G, 
        target_num_parts=target_num_parts,
        verbose=verbose
    )
    partition = decomp_result['partition']
    
    # Step 2: Build partition graph
    if verbose:
        print("\n[Step 2] Building partition connectivity graph...")
    
    partition_graph = build_partition_graph(G, partition)
    inter_edges = compute_inter_partition_edges(G, partition)
    
    if verbose:
        total_inter = sum(len(e) for e in inter_edges.values())
        print(f"  Inter-partition edges: {total_inter}")
        print(f"  Partition graph edges: {partition_graph.number_of_edges()}")
    
    # Step 3: Find optimal partition ordering
    if verbose:
        print("\n[Step 3] Finding optimal partition ordering...")
    
    partition_ordering = optimal_partition_ordering_tsp(G, partition)
    
    if verbose:
        print(f"  Partition order: {partition_ordering}")
    
    # Step 4: Compute interface information
    if verbose:
        print("\n[Step 4] Analyzing partition interfaces...")
    
    interface_info = compute_partition_interface(G, partition)
    
    if verbose:
        for i, info in interface_info.items():
            n_interface = len(info['interface_vertices'])
            n_internal = len(info['internal_vertices'])
            print(f"  Partition {i}: {n_interface} interface, {n_internal} internal vertices")
    
    # Step 5: Generate linear arrangement
    if verbose:
        print("\n[Step 5] Generating linear arrangement...")
    
    linear_order, arrangement_metrics = generate_linear_arrangement(
        G, partition, partition_ordering
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print("MINLA RESULTS")
        print(f"{'='*60}")
        print(f"  Total edge length (MinLA cost): {arrangement_metrics['total_edge_length']}")
        print(f"  Maximum edge length: {arrangement_metrics['max_edge_length']}")
        print(f"  Average edge length: {arrangement_metrics['avg_edge_length']:.2f}")
        print(f"  Number of edges: {arrangement_metrics['num_edges']}")
    
    # Combine all results
    result = {
        **decomp_result,
        'linear_order': linear_order,
        'minla_cost': arrangement_metrics['total_edge_length'],
        'max_edge_length': arrangement_metrics['max_edge_length'],
        'avg_edge_length': arrangement_metrics['avg_edge_length'],
        'partition_ordering': partition_ordering,
        'partition_graph': partition_graph,
        'interface_info': interface_info,
        'inter_partition_edges': inter_edges
    }
    
    return result


# ============================================================================
# Example Usage and Testing
# ============================================================================

def create_distance_hereditary_graph(n: int = 10) -> nx.Graph:
    """
    Create a random CONNECTED distance-hereditary graph (fully decomposable via rank-1 cuts).
    
    Distance-hereditary graphs can be constructed by:
    - Starting with a single vertex
    - Adding pendant vertices (degree 1)
    - Adding true twins (same closed neighborhood)
    - Adding false twins (same open neighborhood)
    
    This version ensures the graph stays connected.
    """
    G = nx.Graph()
    G.add_node(0)
    
    for v in range(1, n):
        existing = list(G.nodes())
        u = np.random.choice(existing)
        
        # Check if u has neighbors (needed for false twin to maintain connectivity)
        u_neighbors = list(G.neighbors(u))
        
        if len(u_neighbors) == 0:
            # u is isolated, can only add pendant or true twin
            operation = np.random.choice(['pendant', 'true_twin'])
        else:
            operation = np.random.choice(['pendant', 'true_twin', 'false_twin'])
        
        if operation == 'pendant':
            # Add pendant vertex attached only to u
            G.add_node(v)
            G.add_edge(v, u)
        elif operation == 'true_twin':
            # Add true twin of u (same closed neighborhood)
            G.add_node(v)
            G.add_edge(v, u)  # True twins are adjacent
            for neighbor in u_neighbors:
                if neighbor != v:
                    G.add_edge(v, neighbor)
        else:  # false_twin
            # Add false twin of u (same open neighborhood, not adjacent to u)
            G.add_node(v)
            for neighbor in u_neighbors:
                G.add_edge(v, neighbor)
    
    return G


def is_distance_hereditary(G: nx.Graph) -> bool:
    """
    Recognize distance-hereditary graphs by reverse construction.

    Repeatedly remove a vertex that is either:
    - isolated or pendant (degree <= 1), or
    - a twin of another vertex (true or false twin: same neighborhood excluding each other)

    If the graph can be reduced to empty, it is distance-hereditary.
    """
    H = G.copy()
    while H.number_of_nodes() > 0:
        removed = False
        nodes = list(H.nodes())

        # remove isolated or pendant vertices first
        for v in nodes:
            if H.degree(v) <= 1:
                H.remove_node(v)
                removed = True
                break

        if removed:
            continue

        # look for twin vertices
        found_twin = False
        for i, v in enumerate(nodes):
            Nv = set(H.neighbors(v))
            for u in nodes[i+1:]:
                Nu = set(H.neighbors(u))
                if (Nv - {u}) == (Nu - {v}):
                    H.remove_node(v)
                    found_twin = True
                    removed = True
                    break
            if found_twin:
                break

        if removed:
            continue

        # if no removable vertex found, not distance-hereditary
        return False

    return True


def example_usage():
    """Demonstrate the universal partitioning algorithm on various graph types."""
    
    print("=" * 70)
    print("UNIVERSAL GRAPH PARTITIONING DEMONSTRATION")
    print("Works for ANY connected simple graph (DH and non-DH)")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Example 1: Distance-Hereditary Graph (will use optimized DH algorithm)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 1: Distance-Hereditary Graph (n=12)")
    print("=" * 70)
    np.random.seed(42)
    G1 = create_distance_hereditary_graph(12)
    result1 = universal_partition(G1, target_num_parts=4, verbose=True)
    print(f"\nPartition: {[sorted(list(p)) for p in result1['partition']]}")
    
    # -------------------------------------------------------------------------
    # Example 2: Complete Graph K_8 (DH - all vertices are twins)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 2: Complete Graph K_8 (Distance-Hereditary)")
    print("=" * 70)
    G2 = nx.complete_graph(8)
    result2 = universal_partition(G2, target_num_parts=4, verbose=True)
    print(f"\nPartition: {[sorted(list(p)) for p in result2['partition']]}")
    
    # -------------------------------------------------------------------------
    # Example 3: Cycle Graph C_10 (NOT Distance-Hereditary for n > 4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 3: Cycle Graph C_10 (NOT Distance-Hereditary)")
    print("=" * 70)
    G3 = nx.cycle_graph(10)
    result3 = universal_partition(G3, target_num_parts=4, verbose=True)
    print(f"\nPartition: {[sorted(list(p)) for p in result3['partition']]}")
    
    # -------------------------------------------------------------------------
    # Example 4: Random Erdos-Renyi Graph (n=15, p=0.5) - likely NOT DH
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 4: Random Erdos-Renyi Graph (n=15, p=0.5)")
    print("=" * 70)
    np.random.seed(123)
    # Generate connected random graph
    attempts = 0
    G4 = None
    while attempts < 100:
        seed_val = np.random.randint(0, 2**31 - 1)
        Gcand = nx.erdos_renyi_graph(15, 0.5, seed=int(seed_val))
        if nx.is_connected(Gcand):
            G4 = Gcand
            break
        attempts += 1
    
    if G4 is not None:
        result4 = universal_partition(G4, target_num_parts=5, verbose=True)
        print(f"\nPartition: {[sorted(list(p)) for p in result4['partition']]}")
    else:
        print("Failed to generate connected graph")
    
    # -------------------------------------------------------------------------
    # Example 5: Grid Graph 4x4 (NOT Distance-Hereditary)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 5: Grid Graph 4x4 (NOT Distance-Hereditary)")
    print("=" * 70)
    G5 = nx.grid_2d_graph(4, 4)
    # Relabel to integers for cleaner output
    G5 = nx.convert_node_labels_to_integers(G5)
    result5 = universal_partition(G5, target_num_parts=4, verbose=True)
    print(f"\nPartition: {[sorted(list(p)) for p in result5['partition']]}")
    
    # -------------------------------------------------------------------------
    # Example 6: Petersen Graph (Famous non-DH graph)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 6: Petersen Graph (NOT Distance-Hereditary)")
    print("=" * 70)
    G6 = nx.petersen_graph()
    result6 = universal_partition(G6, target_num_parts=5, verbose=True)
    print(f"\nPartition: {[sorted(list(p)) for p in result6['partition']]}")
    
    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: Balance Metrics Comparison")
    print("=" * 70)
    results = [
        ("DH Graph (n=12)", result1),
        ("Complete K_8", result2),
        ("Cycle C_10", result3),
        ("ER(15,0.5)", result4 if G4 else None),
        ("Grid 4x4", result5),
        ("Petersen", result6),
    ]
    
    print(f"{'Graph':<20} {'DH?':<5} {'Parts':<6} {'Sizes':<15} {'CV':<8} {'Ratio':<8} {'Rank-1':<8} {'Balanced?':<10}")
    print("-" * 90)
    for name, r in results:
        if r is None:
            continue
        dh = "Yes" if r['is_distance_hereditary'] else "No"
        parts = len(r['partition'])
        sizes_str = str(r['sizes'])[:14]
        cv = f"{r['balance_metrics']['cv']:.3f}"
        ratio = f"{r['balance_metrics']['size_ratio']:.2f}"
        rank1 = f"{r['num_rank1_parts']}/{parts}"
        balanced = "Yes" if r['is_balanced_enough'] else "No"
        print(f"{name:<20} {dh:<5} {parts:<6} {sizes_str:<15} {cv:<8} {ratio:<8} {rank1:<8} {balanced:<10}")
    
    # -------------------------------------------------------------------------
    # MinLA Demonstration
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MINLA INTEGRATION DEMONSTRATION")
    print("Graph Decomposition -> Linear Arrangement Pipeline")
    print("=" * 70)
    
    # Use Grid graph as it's a good MinLA benchmark
    print("\n--- Grid 4x4 MinLA Analysis ---")
    minla_result = decomposition_for_minla(G5, target_num_parts=4, verbose=True)
    
    print(f"\nLinear arrangement: {minla_result['linear_order']}")
    print(f"MinLA cost (total edge length): {minla_result['minla_cost']}")
    print(f"Max edge length: {minla_result['max_edge_length']}")
    print(f"Average edge length: {minla_result['avg_edge_length']:.2f}")
    
    # Compare with a random arrangement
    random_order = list(G5.nodes())
    np.random.shuffle(random_order)
    random_pos = {v: i for i, v in enumerate(random_order)}
    random_cost = sum(abs(random_pos[u] - random_pos[v]) for u, v in G5.edges())
    print(f"\nRandom arrangement cost: {random_cost}")
    print(f"Improvement: {100 * (random_cost - minla_result['minla_cost']) / random_cost:.1f}%")
    
    # Also show Petersen graph MinLA
    print("\n--- Petersen Graph MinLA Analysis ---")
    minla_petersen = decomposition_for_minla(G6, target_num_parts=5, verbose=True)
    
    print(f"\nLinear arrangement: {minla_petersen['linear_order']}")
    print(f"MinLA cost: {minla_petersen['minla_cost']}")


if __name__ == "__main__":
    example_usage()
