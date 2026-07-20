"""
the stabilizer-tableau height function (used by
lib/tableau.py's stabTableau, and therefore by run_optimization()) can
report ONE MORE emitter than the true GF(2) cut-rank for the same graph
and the same vertex ordering.

Run from photonic-graphstate-generator-main/:
    python demo_height_function_bug.py
"""
import networkx as nx
import numpy as np

import rank_width as rw                      # true GF(2) cut-rank
from lib.generate_graph import GraphstateGenerator
from lib.tableau import stabTableau           # actual class used by run_optimization()


def true_lrw(adj, ordering):
    """Ground-truth linear rank-width for this ordering (GF(2) cut-rank max)."""
    n = len(ordering)
    return max(rw._cut_rank(adj, ordering, k) for k in range(n - 1))


def evaluate_fixed(G, ordering, label):
    """No search, no optimization: just evaluate one fixed ordering and
    print both quantities side by side."""
    adj = nx.adjacency_matrix(G).toarray().astype(np.uint8)
    lrw = true_lrw(adj, ordering)

    ordered_adj = adj[np.ix_(ordering, ordering)]
    tableau = GraphstateGenerator.get_tableau_from_adj(ordered_adj)
    n_e = stabTableau(tableau).n_e

    flag = "OK" if n_e == lrw else "MISMATCH"
    print(f"  {label:30s} ordering={ordering}  true_lrw={lrw}  stabTableau.n_e={n_e}  [{flag}]")


# --- Fixed path graph P9, hand-picked orderings (no search/optimization) ---
print("=== Path graph P9, fixed hand-picked orderings ===")
P9 = nx.path_graph(9)   # edges 0-1-2-3-4-5-6-7-8

evaluate_fixed(P9, [0, 1, 2, 3, 4, 5, 6, 7, 8], "natural order (0-1-...-8)")
evaluate_fixed(P9, [0, 1, 2, 3, 4, 6, 5, 7, 8], "single swap (5,6)")
evaluate_fixed(P9, [0, 1, 2, 4, 3, 6, 5, 7, 8], "two swaps (3,4) and (5,6)")


# --- Fixed cycle graph C9, hand-picked orderings (no search/optimization) ---
print("\n=== Cycle graph C9 (0-1-2-...-8-0), fixed hand-picked orderings ===")
C9 = nx.cycle_graph(9)   # edges 0-1-2-...-7-8-0

evaluate_fixed(C9, [0, 1, 2, 3, 4, 5, 6, 7, 8], "natural order (0-1-...-8-0)")
evaluate_fixed(C9, [0, 1, 2, 3, 4, 6, 5, 7, 8], "single swap (5,6)")
evaluate_fixed(C9, [1, 0, 2, 3, 4, 5, 6, 7, 8], "single swap (0,1)")
evaluate_fixed(C9, [0, 1, 2, 3, 4, 5, 6, 8, 7], "single swap (7,8)")
evaluate_fixed(C9, [0, 1, 2, 4, 3, 6, 5, 7, 8], "two swaps (3,4) and (5,6)")
evaluate_fixed(C9, [8, 1, 2, 3, 4, 5, 6, 7, 0], "swap the two ends (0,8)")
evaluate_fixed(C9, [0, 2, 1, 3, 5, 4, 6, 8, 7], "three adjacent-pair swaps")


# --- Fixed star graph, swapping the CENTER label with an OUTER (leaf) label ---
print("\n=== Star graph (center=0, leaves=1..8), fixed hand-picked orderings ===")
Star9 = nx.star_graph(8)   # center node 0, leaves 1..8

evaluate_fixed(Star9, [0, 1, 2, 3, 4, 5, 6, 7, 8], "natural order (center first)")
evaluate_fixed(Star9, [4, 1, 2, 3, 0, 5, 6, 7, 8], "center swapped with leaf 4")
evaluate_fixed(Star9, [8, 1, 2, 3, 4, 5, 6, 7, 0], "center swapped with leaf 8 (last)")
evaluate_fixed(Star9, [1, 0, 2, 3, 4, 5, 6, 7, 8], "center swapped with leaf 1 (adjacent position)")
evaluate_fixed(Star9, [1, 4, 2, 3, 0, 5, 6, 7, 8], "center swapped with leaf 4, then leaves 1&4 swapped")
