"""
Linear Rank-Width Heuristic Solver
===================================

Approximates the Linear Rank-Width (lrw) of a graph G = (V, E).

Pipeline
--------
Phase A - Initialization  : Spectral, RCM, and Min-Degree orderings.
Phase B - Evaluation       : Compute lrw for each candidate ordering via
                             GF(2) matrix rank at every linear cut.
Phase C - Refinement       : Hill-climbing local search around the bottleneck.

Usage
-----
    from rank_width import linear_rank_width

    lrw, best_ordering = linear_rank_width(G)

Or from the command line (reads a graph from an edge-list file):

    python rank_width.py --file graph.edgelist
"""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee


# ---------------------------------------------------------------------------
#  Phase B helpers – GF(2) linear algebra
# ---------------------------------------------------------------------------

def _gf2_rank(matrix: np.ndarray) -> int:
    """Compute the rank of a binary matrix over GF(2) using Gaussian elimination.

    Parameters
    ----------
    matrix : np.ndarray
        A 2-D array whose entries are 0 or 1.

    Returns
    -------
    int
        The GF(2) rank.
    """
    if matrix.size == 0:
        return 0

    # Work on a copy cast to uint8 for speed; we only need mod-2 arithmetic.
    M = matrix.astype(np.uint8, copy=True)
    nrows, ncols = M.shape
    rank = 0
    pivot_col = 0

    for row in range(nrows):
        if pivot_col >= ncols:
            break

        # Find a pivot in the current column from 'row' downward.
        found = False
        for k in range(row, nrows):
            if M[k, pivot_col]:
                found = True
                if k != row:
                    # Swap rows
                    M[[row, k]] = M[[k, row]]
                break

        if not found:
            pivot_col += 1
            # Retry the same row with the next column
            # (implemented via a while loop below instead of recursion)
            continue

        # Eliminate all other 1s in this column
        for k in range(nrows):
            if k != row and M[k, pivot_col]:
                M[k] ^= M[row]  # XOR = addition in GF(2)

        rank += 1
        pivot_col += 1

    return rank


def _gf2_rank_bitpacked(matrix: np.ndarray) -> int:
    """GF(2) rank using bit-packed rows for large matrices (faster).

    Each row is stored as a Python int treated as a bit-vector.
    """
    if matrix.size == 0:
        return 0

    M = matrix.astype(np.uint8, copy=True)
    nrows, ncols = M.shape

    # Pack each row into a single Python int
    rows = []
    for i in range(nrows):
        val = 0
        for j in range(ncols):
            if M[i, j]:
                val |= (1 << j)
        rows.append(val)

    rank = 0
    for col in range(ncols):
        mask = 1 << col
        # Find pivot
        pivot = -1
        for i in range(rank, nrows):
            if rows[i] & mask:
                pivot = i
                break
        if pivot == -1:
            continue
        # Swap pivot row into position
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        # Eliminate
        for i in range(nrows):
            if i != rank and (rows[i] & mask):
                rows[i] ^= rows[rank]
        rank += 1

    return rank


def gf2_rank(matrix: np.ndarray) -> int:
    """Dispatch to the most efficient GF(2) rank routine."""
    if matrix.size == 0:
        return 0
    nrows, ncols = matrix.shape
    # For wider matrices bit-packing is faster
    if ncols > 64:
        return _gf2_rank_bitpacked(matrix)
    return _gf2_rank(matrix)


# ---------------------------------------------------------------------------
#  Phase B – Evaluation
# ---------------------------------------------------------------------------

def _cut_rank(adj: np.ndarray, ordering: list[int], cut_index: int) -> int:
    """ Compute the GF(2) rank of the cut sub-matrix at position *cut_index*.
        The cut separates ordering[:cut_index+1]  (left) from     ordering[cut_index+1:]  (right).
    """
    left = ordering[: cut_index + 1]
    right = ordering[cut_index + 1 :]
    sub = adj[np.ix_(left, right)]
    return gf2_rank(sub)


def evaluate_ordering(graph: nx.Graph, ordering: list[int]) -> tuple[int, int]:
    """Evaluate the linear rank-width of a specific vertex ordering.

    Parameters
    ----------
    graph : nx.Graph
        Input graph (nodes labelled 0 .. n-1).
    ordering : list[int]
        A permutation of the vertex set.

    Returns
    -------
    lrw : int
        The maximum cut-rank over all n-1 cuts.
    bottleneck : int
        The cut index that realises the maximum rank.
    """
    n = len(ordering)
    if n <= 1:
        return 0, 0

    adj = nx.adjacency_matrix(graph).toarray().astype(np.uint8)

    max_rank = 0
    bottleneck = 0
    for i in range(n - 1):
        r = _cut_rank(adj, ordering, i)
        if r > max_rank:
            max_rank = r
            bottleneck = i

    return max_rank, bottleneck


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
#  Phase C – Refinement (Hill-Climbing Local Search)
# ---------------------------------------------------------------------------

def _swap_and_evaluate(
    adj: np.ndarray,
    ordering: list[int],
    i: int,
    j: int,
) -> tuple[int, int]:
    """Swap positions *i* and *j* in *ordering* and return (max_rank, bottleneck)."""
    ordering[i], ordering[j] = ordering[j], ordering[i]
    n = len(ordering)
    max_rank = 0
    bottleneck = 0
    for k in range(n - 1):
        r = _cut_rank(adj, ordering, k)
        if r > max_rank:
            max_rank = r
            bottleneck = k
    return max_rank, bottleneck


def hill_climb(
    graph: nx.Graph,
    ordering: list[int],
    current_lrw: int,
    current_bottleneck: int,
    max_iterations: int = 200,
    window: int = 5,
    seed: Optional[int] = None,
) -> tuple[list[int], int, int]:
    """Hill-climbing refinement around the bottleneck cut.

    Strategy
    --------
    1.  Identify the bottleneck cut index k.
    2.  Try all pairwise swaps within a window of size window around k.
    3.  Accept the first improving swap; update bottleneck and repeat.
    4.  If no improving swap is found in the window, try random swaps.
    5.  Stop after max_iterations non-improving rounds.

    Parameters
    ----------
    graph : nx.Graph
    ordering : list[int]
        Current best ordering (will be mutated in-place on improvement).
    current_lrw : int
    current_bottleneck : int
    max_iterations : int
    window : int
        Half-width of the neighbourhood around the bottleneck.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ordering : list[int]
    best_lrw : int
    best_bottleneck : int
    """
    rng = random.Random(seed)
    adj = nx.adjacency_matrix(graph).toarray().astype(np.uint8)
    n = len(ordering)

    best_lrw = current_lrw
    best_bottleneck = current_bottleneck
    no_improve = 0

    for _iteration in range(max_iterations):
        improved = False

        # --- Deterministic neighbourhood around the bottleneck ---
        lo = max(0, best_bottleneck - window)
        hi = min(n - 1, best_bottleneck + window + 1)
        neighbourhood = list(range(lo, hi))

        for i in range(len(neighbourhood)):
            for j in range(i + 1, len(neighbourhood)):
                pi, pj = neighbourhood[i], neighbourhood[j]
                # Try swap
                ordering[pi], ordering[pj] = ordering[pj], ordering[pi]

                # Evaluate only the cuts that could have changed
                # (conservative: re-evaluate all for correctness)
                new_lrw = 0
                new_bn = 0
                for k in range(n - 1):
                    r = _cut_rank(adj, ordering, k)
                    if r > new_lrw:
                        new_lrw = r
                        new_bn = k

                if new_lrw < best_lrw:
                    best_lrw = new_lrw
                    best_bottleneck = new_bn
                    improved = True
                    break  # accept first improvement
                else:
                    # Undo swap
                    ordering[pi], ordering[pj] = ordering[pj], ordering[pi]

            if improved:
                break

        if improved:
            no_improve = 0
            continue

        # --- Random perturbation (escape local minimum) ---
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            no_improve += 1
            continue

        ordering[i], ordering[j] = ordering[j], ordering[i]
        new_lrw = 0
        new_bn = 0
        for k in range(n - 1):
            r = _cut_rank(adj, ordering, k)
            if r > new_lrw:
                new_lrw = r
                new_bn = k

        if new_lrw < best_lrw:
            best_lrw = new_lrw
            best_bottleneck = new_bn
            no_improve = 0
        else:
            ordering[i], ordering[j] = ordering[j], ordering[i]
            no_improve += 1

        if no_improve >= max_iterations // 2:
            break  # early stopping

    return ordering, best_lrw, best_bottleneck


# ---------------------------------------------------------------------------
#  Full Pipeline
# ---------------------------------------------------------------------------

def linear_rank_width(
    graph: nx.Graph,
    refine: bool = True,
    max_hill_climb_iter: int = 300,
    hill_climb_window: int = 5,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> tuple[int, list]:
    """Approximate the Linear Rank-Width of graph.

    Parameters
    ----------
    graph : nx.Graph
        The input graph (any node labels).
    refine : bool
        Whether to run the Phase C hill-climbing refinement.
    max_hill_climb_iter : int
        Maximum iterations for the hill-climbing phase.
    hill_climb_window : int
        Half-window around the bottleneck for local swaps.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress information.

    Returns
    -------
    lrw : int
        The (heuristic) linear rank-width.
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
            print("Phase C: Hill-climbing refinement ...")
        best_ordering, best_lrw, best_bottleneck = hill_climb(
            G_int,
            best_ordering,
            best_lrw,
            best_bottleneck,
            max_iterations=max_hill_climb_iter,
            window=hill_climb_window,
            seed=seed,
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
#  CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(
        description="Approximate the Linear Rank-Width of a graph."
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to an edge-list file readable by networkx.read_edgelist().",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the Phase C hill-climbing refinement.",
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=300,
        help="Max hill-climbing iterations (default: 300).",
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=5,
        help="Hill-climbing window half-width (default: 5).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        G = nx.read_edgelist(args.file, nodetype=int)
    except Exception:
        G = nx.read_edgelist(args.file)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    t0 = time.perf_counter()
    lrw, ordering = linear_rank_width(
        G,
        refine=not args.no_refine,
        max_hill_climb_iter=args.iterations,
        hill_climb_window=args.window,
        seed=args.seed,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0

    print(f"Linear Rank-Width (approx): {lrw}")
    print(f"Time: {elapsed:.3f}s")
    if args.verbose:
        print(f"Ordering: {ordering}")


if __name__ == "__main__":
    _cli()
