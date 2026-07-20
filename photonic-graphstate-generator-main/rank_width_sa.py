"""
Linear Rank-Width Heuristic Solver  (Simulated Annealing)
==========================================================

Approximates the Linear Rank-Width (lrw) of a graph G = (V, E).

Pipeline
--------
Phase A Initialization  : Spectral, RCM, and Min-Degree orderings.
Phase B Evaluation       : Compute lrw for each candidate ordering via
                             GF(2) matrix rank at every linear cut.
Phase C Refinement       : Simulated annealing with swap, segment-reversal,
                             and vertex-relocation moves, Metropolis acceptance,
                             geometric cooling, and optional reheating.

Usage
-----
    from rank_width_sa import linear_rank_width_sa

    lrw, best_ordering = linear_rank_width_sa(G)

Or from the command line (reads a graph from an edge-list file):

    python rank_width_sa.py --file graph.edgelist
"""

from __future__ import annotations

import math
import random
from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee

from optimization_script import run_optimization
from optgraphstate import GraphState
from lib.generate_graph import GraphstateGenerator
from utils.ufuncs import heightfunction
from rank_width import (
    _gf2_rank,
    _gf2_rank_bitpacked,
    gf2_rank,
    _cut_rank,
)

# ---------------------------------------------------------------------------
#  Phase B helpers – GF(2) linear algebra
# ---------------------------------------------------------------------------
#
# NOTE: _gf2_rank / _gf2_rank_bitpacked / gf2_rank / _cut_rank used to be
# duplicated verbatim in this module. They are now imported from
# rank_width.py so there is a single implementation to maintain (see
# HEIGHT_FUNCTION_BUG_README.md for the bug this duplication previously
# caused to drift out of sync).


# ---------------------------------------------------------------------------
#  Phase B – Evaluation
# ---------------------------------------------------------------------------

def evaluate_ordering(graph: nx.Graph, ordering: list[int]) -> tuple[int, int]:
    """Evaluate the number of emitters for a specific vertex ordering.

    Uses the stabilizer-tableau height function: the number of emitters
    equals ``max(h)`` and, at optimality, equals the linear rank-width.

    Parameters
    ----------
    graph : nx.Graph
        Input graph (nodes labelled 0 .. n-1).
    ordering : list[int]
        A permutation of the vertex set.

    Returns
    -------
    n_emitters : int
        ``max(h)`` the number of emitters required for this ordering.
    bottleneck : int
        The step index where the height function is maximised.
    """
    n = len(ordering)
    if n <= 1:
        return 0, 0

    adj = nx.adjacency_matrix(graph).toarray().astype(np.uint8)
    return _full_evaluate(adj, ordering)


# ---------------------------------------------------------------------------
#  Phase A – Initialization (Population Generation)
# ---------------------------------------------------------------------------

def _relabel_to_int(graph: nx.Graph) -> tuple[nx.Graph, dict, dict]:
    """Relabel graph nodes to consecutive integers 0..n-1. Sometimes networkx has different labeling schemes.
        For grids for example it uses (x,y) tuples which are inconvenient for our matrix-based computations. 
        This function also returns the mapping dictionaries to convert between original and integer labels.
    

    Returns
    -------
    G_int : nx.Graph
    old_to_new : dict   original_node -> int
    new_to_old : dict   int -> original_node
    """
    nodes = list(graph.nodes())
    old_to_new = {v: i for i, v in enumerate(nodes)}
    new_to_old = {i: v for i, v in enumerate(nodes)}
    G_int = nx.relabel_nodes(graph, old_to_new)
    return G_int, old_to_new, new_to_old


def spectral_ordering(graph: nx.Graph) -> list[int]:
    """Sort vertices by the Fiedler vector (2nd smallest eigenvector of
    the graph Laplacian).  Captures global geometric structure.
    """
    n = graph.number_of_nodes()
    if n <= 2:
        return list(range(n))

    L = nx.laplacian_matrix(graph).toarray().astype(float)
    eigvals, eigvecs = np.linalg.eigh(L)

    # The Fiedler vector is the eigenvector for the 2nd-smallest eigenvalue.
    fiedler = eigvecs[:, 1]
    ordering = list(np.argsort(fiedler))
    return ordering


def rcm_ordering(graph: nx.Graph) -> list[int]:
    """Reverse Cuthill-McKee ordering"""
    n = graph.number_of_nodes()
    if n <= 2:
        return list(range(n))

    adj_sparse = nx.adjacency_matrix(graph)
    perm = reverse_cuthill_mckee(adj_sparse)
    return list(perm)


def min_degree_ordering(graph: nx.Graph) -> list[int]:
    """Greedy ordering that always appends the vertex of minimum degree
    among unvisited nodes (ties broken by connectivity to already-placed
    vertices, then arbitrarily).
    """
    n = graph.number_of_nodes()
    if n <= 2:
        return list(range(n))

    remaining = set(range(n))
    ordering: list[int] = []
    placed: set[int] = set()

    for _ in range(n):
        # Pick vertex with minimum degree in the subgraph induced by remaining vertices. Break ties by maximum adjacency to placed set.
        best_v = min(
            remaining,
            key=lambda v: (
                sum(1 for u in graph.neighbors(v) if u in remaining),
                -sum(1 for u in graph.neighbors(v) if u in placed),
            ),
        )
        ordering.append(best_v)
        placed.add(best_v)
        remaining.remove(best_v)

    return ordering


def generate_initial_orderings(graph: nx.Graph) -> list[list[int]]:
    """Phase A: produce three structurally diverse candidate orderings."""
    candidates = [
        spectral_ordering(graph),
        rcm_ordering(graph),
        min_degree_ordering(graph),
    ]
    return candidates


# ---------------------------------------------------------------------------
#  Phase C – Refinement (Simulated Annealing)
# ---------------------------------------------------------------------------

def _full_evaluate(adj: np.ndarray, ordering: list[int]) -> tuple[int, int]:
    """Evaluate the number of emitters (max of height function) for an ordering.

    For graph states the number of emitters equals ``max(h)`` where h
    is the height function of the stabilizer tableau.  This is the
    correct cost to minimise: ``min over a of max(h(a)) = lrw(G)``.

    Returns
    -------
    n_emitters : int
        ``max(h)`` the number of emitters required.
    bottleneck : int
        The step index where the height function is maximised.
    """
    n = len(ordering)
    # Build adjacency matrix in the given ordering
    ordered_adj = adj[np.ix_(ordering, ordering)]
    tableau = GraphstateGenerator.get_tableau_from_adj(ordered_adj)
    h = heightfunction(tableau, n)
    n_emitters = int(max(h))
    bottleneck = int(np.argmax(h))
    return n_emitters, bottleneck


def _neighbour_swap(ordering: list[int], rng: random.Random) -> tuple[list[int], tuple]:
    """Generate a neighbour by swapping two random positions.

    Returns the new ordering and an undo token (i, j).
    """
    n = len(ordering)
    i = rng.randint(0, n - 1)
    j = rng.randint(0, n - 1)
    while j == i:
        j = rng.randint(0, n - 1)
    ordering[i], ordering[j] = ordering[j], ordering[i]
    return ordering, ("swap", i, j)


def _neighbour_reverse(ordering: list[int], rng: random.Random) -> tuple[list[int], tuple]:
    """Generate a neighbour by reversing a random sub-segment.

    Returns the new ordering and an undo token.
    """
    n = len(ordering)
    i = rng.randint(0, n - 2)
    max_len = min(n - i, 8)          # cap segment length for locality
    seg_len = rng.randint(2, max_len)
    j = i + seg_len
    old_segment = ordering[i:j][:]    # save for undo
    ordering[i:j] = ordering[i:j][::-1]
    return ordering, ("reverse", i, j, old_segment)


def _neighbour_relocate(ordering: list[int], rng: random.Random) -> tuple[list[int], tuple]:
    """Generate a neighbour by relocating a single vertex to a new position.

    Returns the new ordering and an undo token.
    """
    n = len(ordering)
    src = rng.randint(0, n - 1)
    dst = rng.randint(0, n - 1)
    while dst == src:
        dst = rng.randint(0, n - 1)
    v = ordering[src]
    old_ordering = ordering[:]        # save for undo
    ordering.pop(src)
    ordering.insert(dst, v)
    return ordering, ("relocate", old_ordering)


def _undo_move(ordering: list[int], token: tuple) -> list[int]:
    """Undo the perturbation described by *token* (in-place)."""
    kind = token[0]
    if kind == "swap":
        _, i, j = token
        ordering[i], ordering[j] = ordering[j], ordering[i]
    elif kind == "reverse":
        _, i, j, old_segment = token
        ordering[i:j] = old_segment
    elif kind == "relocate":
        _, old_ordering = token
        ordering[:] = old_ordering
    return ordering


def simulated_annealing(
    graph: nx.Graph,
    ordering: list[int],
    current_lrw: int,
    current_bottleneck: int,
    *,
    T_start: float = 4.0,
    T_min: float = 0.01,
    alpha: float = 0.97,
    steps_per_temp: int = 50,
    reheat_interval: int = 0,
    reheat_factor: float = 2.0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> tuple[list[int], int, int]:
    """Simulated-annealing refinement of a vertex ordering to minimise
    the number of emitters (= linear rank-width at optimality).

    Strategy
    --------
    The cost function is ``max(h)`` the height-function maximum of the
    stabilizer tableau which equals the number of emitters needed to
    generate the graph state under the given ordering.  At optimality
    this coincides with the linear rank-width.

    Starting from the best ordering found in Phases A-B, the algorithm
    explores the neighbourhood by applying random perturbations (pairwise
    swaps, segment reversals, and vertex relocations) and accepting or
    rejecting each move via the Metropolis criterion:

        accept if  Delta < 0  (improving)
        accept if  exp(-Delta / T) > uniform(0, 1)   (worsening)

    The temperature T is cooled geometrically from T_start to T_min
    by factor alpha after every steps_per_temp perturbations.  An
    optional periodic reheat avoids premature convergence.

    Parameters
    ----------
    graph : nx.Graph
        Input graph (nodes 0 .. n-1).
    ordering : list[int]
        Initial ordering (will be mutated).
    current_lrw : int
        Number of emitters (max(h)) of the initial ordering.
    current_bottleneck : int
        Step index realising the maximum of the height function.
    T_start : float
        Initial temperature.
    T_min : float
        Minimum temperature (termination condition).
    alpha : float
        Geometric cooling factor  (0 < alpha < 1).
    steps_per_temp : int
        Number of perturbation trials at each temperature level.
    reheat_interval : int
        If > 0, multiply T by reheat_factor every this many cooling
        steps.  Set to 0 to disable reheating.
    reheat_factor : float
        Multiplicative factor applied on reheat.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print periodic progress information.

    Returns
    -------
    best_ordering : list[int]
    best_lrw : int
    best_bottleneck : int
    """
    rng = random.Random(seed)
    adj = nx.adjacency_matrix(graph).toarray().astype(np.uint8)
    n = len(ordering)

    # --- SA state ---
    cur_lrw = current_lrw
    cur_bn = current_bottleneck

    best_lrw = cur_lrw
    best_bn = cur_bn
    best_ordering = ordering[:]

    # Neighbourhood move selection (weighted)
    move_funcs = [_neighbour_swap, _neighbour_reverse, _neighbour_relocate]
    move_weights = [0.50, 0.30, 0.20]       # swap is cheapest to undo

    T = T_start
    cooling_step = 0
    total_accepted = 0
    total_tried = 0

    while T > T_min:
        for _step in range(steps_per_temp):
            total_tried += 1

            # --- Pick a random move ---
            r = rng.random()
            cum = 0.0
            move_idx = 0
            for idx, w in enumerate(move_weights):
                cum += w
                if r < cum:
                    move_idx = idx
                    break
            move_fn = move_funcs[move_idx]

            # Apply perturbation
            ordering, undo_token = move_fn(ordering, rng)

            # Evaluate new ordering
            new_lrw, new_bn = _full_evaluate(adj, ordering)
            delta = new_lrw - cur_lrw

            # --- Metropolis acceptance ---
            accept = False
            if delta <= 0:
                accept = True
            else:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    accept = True

            if accept:
                total_accepted += 1
                cur_lrw = new_lrw
                cur_bn = new_bn

                # Track global best
                if cur_lrw < best_lrw:
                    best_lrw = cur_lrw
                    best_bn = cur_bn
                    best_ordering = ordering[:]
                    if verbose:
                        print(
                            f"  SA  T={T:.4f}  new best lrw={best_lrw}  "
                            f"(step {total_tried})"
                        )
                    # Early exit if optimal
                    if best_lrw == 0:
                        return best_ordering, best_lrw, best_bn
            else:
                # Reject – undo the move
                ordering = _undo_move(ordering, undo_token)

        # --- Cool down ---
        T *= alpha
        cooling_step += 1

        # --- Optional reheat ---
        if reheat_interval > 0 and cooling_step % reheat_interval == 0:
            T = min(T * reheat_factor, T_start)
            if verbose:
                print(f"  SA  reheat → T={T:.4f}")

    if verbose:
        accept_rate = total_accepted / max(total_tried, 1) * 100
        print(
            f"  SA finished: {total_tried} trials, "
            f"{accept_rate:.1f}% accepted, best lrw={best_lrw}"
        )

    return best_ordering, best_lrw, best_bn


# ---------------------------------------------------------------------------
#  Full Pipeline
# ---------------------------------------------------------------------------

def linear_rank_width_sa(
    graph: nx.Graph,
    refine: bool = True,
    T_start: float = 4.0,
    T_min: float = 0.01,
    alpha: float = 0.97,
    steps_per_temp: int = 50,
    reheat_interval: int = 0,
    reheat_factor: float = 2.0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> tuple[int, list]:
    """Approximate the Linear Rank-Width of graph.

    The cost function used by the SA is the height function of the
    stabilizer tableau (i.e. the number of emitters required for photonic
    graph-state generation).  For graph states the minimum of max(h)
    over all orderings equals the linear rank-width.

    Parameters
    ----------
    graph : nx.Graph
        The input graph (any node labels).
    refine : bool
        Whether to run the Phase C simulated-annealing refinement.
    T_start : float
        Initial SA temperature.
    T_min : float
        Minimum SA temperature (termination condition).
    alpha : float
        Geometric cooling factor (0 < alpha < 1).
    steps_per_temp : int
        Number of perturbation trials at each temperature level.
    reheat_interval : int
        If > 0, multiply T by reheat_factor every this many cooling steps.
        Set to 0 to disable reheating.
    reheat_factor : float
        Multiplicative factor applied on reheat.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress information.

    Returns
    -------
    lrw : int
        The (heuristic) linear rank-width (= number of emitters).
    best_ordering : list
        The vertex ordering achieving this width (in original node labels).
    """
    if graph.number_of_nodes() == 0:
        return 0, []
    if graph.number_of_nodes() == 1:
        return 0, list(graph.nodes())

    # --- Relabel to 0..n-1 ---
    G_int, old_to_new, new_to_old = _relabel_to_int(graph)
    n = G_int.number_of_nodes()

    # --- Phase A: Generate initial orderings ---
    if verbose:
        print("Phase A: Generating initial orderings ...")
    candidates = generate_initial_orderings(G_int)

    # --- Phase B: Evaluate each candidate ---
    if verbose:
        print("Phase B: Evaluating candidates ...")

    best_lrw = n  # upper bound
    best_ordering = candidates[0]
    best_bottleneck = 0
    candidate_names = ["Spectral", "RCM", "MinDegree"]

    for name, ordering in zip(candidate_names, candidates):
        lrw_val, bn = evaluate_ordering(G_int, ordering)
        if verbose:
            print(f"  {name:12s}  lrw = {lrw_val}  (bottleneck at cut {bn})")
        if lrw_val < best_lrw:
            best_lrw = lrw_val
            best_ordering = list(ordering)
            best_bottleneck = bn

    if verbose:
        print(f"Best after Phase B: lrw = {best_lrw}")

    # --- Phase C: Refinement ---
    if refine and best_lrw > 0:
        if verbose:
            print("Phase C: Simulated-annealing refinement ...")
        best_ordering, best_lrw, best_bottleneck = simulated_annealing(
            G_int,
            best_ordering,
            best_lrw,
            best_bottleneck,
            T_start=T_start,
            T_min=T_min,
            alpha=alpha,
            steps_per_temp=steps_per_temp,
            reheat_interval=reheat_interval,
            reheat_factor=reheat_factor,
            seed=seed,
            verbose=verbose,
        )
        if verbose:
            print(f"Best after Phase C: lrw = {best_lrw}")

    # --- Map ordering back to original labels ---
    original_ordering = [new_to_old[v] for v in best_ordering]

    return best_lrw, original_ordering


# ---------------------------------------------------------------------------
#  Convenience helpers
# ---------------------------------------------------------------------------

def lrw_of_ordering(graph: nx.Graph, ordering: list) -> int:
    """Compute the linear rank-width of a specific ordering.

    ordering may use the original node labels of graph.
    The ordering must be a permutation of graph.nodes().
    """
    # Build adjacency matrix with rows/columns in the given ordering
    n = len(ordering)
    adj = nx.adjacency_matrix(graph, nodelist=ordering).toarray().astype(np.uint8)
    # With this nodelist the ordering is simply [0, 1, ..., n-1]
    identity_order = list(range(n))
    max_rank = 0
    for i in range(n - 1):
        left = identity_order[: i + 1]
        right = identity_order[i + 1 :]
        sub = adj[np.ix_(left, right)]
        r = gf2_rank(sub)
        if r > max_rank:
            max_rank = r
    return max_rank


# ---------------------------------------------------------------------------
#  Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Demo: known graph families ---
    print("=== Linear Rank-Width SA Demo ===\n")

    # graphs = [
    #     ("Path P6", nx.path_graph(6)),
    #     ("Cycle C8", nx.cycle_graph(8)),
    #     ("Complete K5", nx.complete_graph(5)),
    #     ("Petersen", nx.petersen_graph()),
    #     ("Star S5", nx.star_graph(5)),
    # ]

    # for name, G in graphs:
    #     lrw, ordering = linear_rank_width_sa(G, verbose=True)
    #     print(f"{name}:  lrw = {lrw}\n")

    print("=== Sanity tests for repeater graphs ===\n")
    prms = 12 #(4,4,4)
    def edge_reduction_func(input_graph):
        our_graph_original = input_graph.copy()
        # Create a mapping dictionary: (0, 0) -> 0, (0, 1) -> 1, ..., (2, 2) -> 8
        mapping = {node: i for i, node in enumerate(our_graph_original.nodes())}

        # Create a new graph with integer labels
        our_graph = nx.relabel_nodes(our_graph_original, mapping)
        graph , circ, statistics = run_optimization(our_graph, edge_reduction=True, minLA=False, opt_CNOT= False)
        return graph
    
    diff_array = []
    emitter_array = []
    cnot_array = []
    # for prms in range(2, maximum+1):
    print(f'----------------- START for {prms} -----------------')
    # Create the graph state for the repeater
    repeater_graph = GraphState(shape='repeater', prms=prms)
    igraph_repeater = repeater_graph.graph
    G = igraph_repeater.to_networkx()

    G = edge_reduction_func(G)

    #### Simulation Set up ####
    alpha_sim = 0.995
    steps_per_temp_sim = 300
    T_start_sim = 7.0
    # reheat_interval_sim = 30
    # reheat_factor_sim = 2.0
    cost, final_ordering = linear_rank_width_sa(G, alpha=alpha_sim, steps_per_temp=steps_per_temp_sim, T_start=T_start_sim, verbose=True)
    print(f"Ordering (using rank width) for rgh with m = {prms} found (lrw={cost})")

    # Visualize and Test with rank_width ordering
    ordered_graph = nx.relabel_nodes(G, {old: new for new, old in enumerate(final_ordering)})
    opt_graph, circuit, stats = run_optimization(ordered_graph, edge_reduction=False, minLA=False, opt_CNOT=True)


    print(f"  Emitters: {stats[0]}, CNOTs: {stats[1]}")
    diff = stats[0] - cost
    diff_array.append(diff)
    emitter_array.append(stats[0])
    cnot_array.append(stats[1])
    print(f"Difference array (emitters - lrw): {diff_array}")
    print(f"Emitter counts: {emitter_array}")
    print(f"CNOT counts: {cnot_array}")

