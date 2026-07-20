"""
Regression tests for the GF(2) rank / linear rank-width fix.

These cover the bug described in HEIGHT_FUNCTION_BUG_README.md:
`_gf2_rank()` in rank_width.py (and its duplicate in rank_width_sa.py) used to
silently under-count the GF(2) rank of a binary matrix whenever a column had
no available pivot, because the retry logic was tied to the outer
`for row in range(nrows)` loop instead of an independent pivot counter.
"""
import random

import networkx as nx
import numpy as np
import pytest

from rank_width import (
    _gf2_rank as rw_gf2_rank,
    _gf2_rank_bitpacked as rw_gf2_rank_bitpacked,
    gf2_rank as rw_gf2_rank_dispatch,
    linear_rank_width,
)
from rank_width_sa import (
    _gf2_rank as rwsa_gf2_rank,
    _gf2_rank_bitpacked as rwsa_gf2_rank_bitpacked,
    gf2_rank as rwsa_gf2_rank_dispatch,
    linear_rank_width_sa,
)
from path_clustering import path_clustering_lrw


def _reference_gf2_rank(matrix: np.ndarray) -> int:
    """Independent, straightforward column-major GF(2) rank used as ground truth."""
    M = matrix.astype(np.uint8).copy()
    nrows, ncols = M.shape
    rank = 0
    for col in range(ncols):
        pivot = None
        for r in range(rank, nrows):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(nrows):
            if r != rank and M[r, col]:
                M[r] ^= M[rank]
        rank += 1
    return rank


class TestGF2RankCorrectness:
    """`_gf2_rank` must match an independent reference and the bit-packed variant."""

    def test_minimal_reproducer_single_row_with_leading_zero_column(self):
        # This is the smallest matrix that triggered the historical bug:
        # a leading all-zero column caused a row to be skipped.
        M = np.array([[0, 1]], dtype=np.uint8)
        assert rw_gf2_rank(M) == 1
        assert rwsa_gf2_rank(M) == 1

    @pytest.mark.parametrize("seed", range(20))
    def test_random_matrices_match_reference(self, seed):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        for _ in range(50):
            nrows = rng.randint(1, 10)
            ncols = rng.randint(1, 10)
            M = np_rng.randint(0, 2, size=(nrows, ncols)).astype(np.uint8)
            ref = _reference_gf2_rank(M)
            assert rw_gf2_rank(M) == ref
            assert rwsa_gf2_rank(M) == ref
            assert rw_gf2_rank_bitpacked(M) == ref
            assert rwsa_gf2_rank_bitpacked(M) == ref
            assert rw_gf2_rank_dispatch(M) == ref
            assert rwsa_gf2_rank_dispatch(M) == ref


class TestObjectiveConsistencyAcrossAlgorithms:
    """After the fix, rank_width.py's cut-rank and the stabilizer height function
    (used by rank_width_sa.py / path_clustering.py) must agree on every cut, for
    every ordering -- not just at the optimum."""

    @pytest.mark.parametrize("seed", range(20))
    def test_cut_rank_matches_height_function_profile(self, seed):
        from lib.generate_graph import GraphstateGenerator
        from utils.ufuncs import heightfunction
        import rank_width as rw

        rng = random.Random(seed)
        n = rng.randint(5, 14)
        p = rng.uniform(0.2, 0.6)
        G = nx.gnp_random_graph(n, p, seed=seed)
        tries = 0
        while not nx.is_connected(G):
            tries += 1
            G = nx.gnp_random_graph(n, min(1.0, p + 0.05 * tries), seed=seed + tries)

        ordering = list(range(n))
        rng.shuffle(ordering)
        adj = nx.adjacency_matrix(G).toarray().astype(np.uint8)

        cutranks = [rw._cut_rank(adj, ordering, k) for k in range(n - 1)]

        ordered_adj = adj[np.ix_(ordering, ordering)]
        tab = GraphstateGenerator.get_tableau_from_adj(ordered_adj)
        h = list(heightfunction(tab, n))

        # h[i] (split sizes i, n-i) should equal cutranks[i-1] for i = 1..n-1
        for i in range(1, n):
            assert int(h[i]) == cutranks[i - 1]


class TestKnownGraphFamilies:
    """Minimum emitter counts for graph families with known linear rank-width,
    checked across all three heuristic algorithms."""

    CASES = [
        ("cycle", nx.cycle_graph(9), 2),
        ("star", nx.star_graph(8), 1),
        ("complete", nx.complete_graph(6), 1),
        ("path", nx.path_graph(9), 1),
    ]

    @pytest.mark.parametrize("name,graph,expected", CASES)
    def test_hill_climbing(self, name, graph, expected):
        lrw, _ = linear_rank_width(graph, seed=0)
        assert lrw == expected

    @pytest.mark.parametrize("name,graph,expected", CASES)
    def test_height_function_sa(self, name, graph, expected):
        lrw, _ = linear_rank_width_sa(graph, seed=0)
        assert lrw == expected

    @pytest.mark.parametrize("name,graph,expected", CASES)
    def test_path_clustering(self, name, graph, expected):
        lrw, _ = path_clustering_lrw(graph, seed=0)
        assert lrw == expected
