# `_gf2_rank()` bug: found, fixed, and independently re-verified

## Problem description

`rank_width.py`'s `_gf2_rank()` had a Gaussian-elimination bug that silently undercounted GF(2)
matrix rank. This fed `hill_climbing`'s cost function, so it could report a linear rank-width
(and consequently emitter count) that's artificially *low* — not a real search-quality advantage, just a
measurement artifact. The bug affected approx. 23% of random small matrices and approx. 8% of random graphs.

The stabilizer-tableau height function (used by `height_function_sa` and `path_clustering`) was
**not** affected — it's confirmed correct against an independent reference.

**The paper's reported numbers are correct based on the ordering the algorithms yield.** This is a bug in the cost function used by the `hill_climbing` algorithm specifically.

## Root cause

`_gf2_rank()` conflates two things that should be independent: the row it's placing a pivot into,
and the row it's currently testing. Both were driven by one `for row in range(nrows)` loop:

```python
for row in range(nrows):
    ...
    if not found:
        pivot_col += 1
        continue   # BUG: advances `row` too (for-loop semantics), even though
                   # the comment above says it should retry the SAME row
    ...
```

When a column has no pivot available, correct Gaussian elimination should advance only the
column and retry the same row. Because the retry uses `continue` inside a `for row` loop, the
loop's automatic increment silently skips that row's chance to become a pivot — so rank only ever
comes out too *low*, never too high.

Note: `_gf2_rank_bitpacked()` (auto-used for matrices with >64 columns) does **not** have this
bug — it already uses two independent counters. Most graphs we tested have less than 64 vertices
per cut, so the buggy path is the one actually used even if the graphs are larger than 64 vertices.

## Fix

Rewrote `_gf2_rank()` in both `rank_width.py` and `rank_width_sa.py` to use a column-major loop
with a separate `rank` counter — mirroring the already-correct structure of
`_gf2_rank_bitpacked()`:

```python
def _gf2_rank(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    M = matrix.astype(np.uint8, copy=True)
    nrows, ncols = M.shape
    rank = 0
    for col in range(ncols):
        if rank >= nrows:
            break
        pivot = None
        for k in range(rank, nrows):
            if M[k, col]:
                pivot = k
                break
        if pivot is None:
            continue
        if pivot != rank:
            M[[rank, pivot]] = M[[pivot, rank]]
        for k in range(nrows):
            if k != rank and M[k, col]:
                M[k] ^= M[rank]
        rank += 1
    return rank
```

`path_clustering.py` needed no direct edit — it imports `gf2_rank` from `rank_width_sa.py` and
inherits the fix.

Added `tests/test_gf2_rank_and_height.py` as a permanent regression suite (53 tests).

## Verification

Two independent passes (original fix + a from-scratch re-audit with fresh seeds) agree:

| Check | Result (buggy, before) | Result (fixed, now) |
|---|---|---|
| `_gf2_rank` vs. reference, random binary matrices | 463/2000 mismatches (approx. 23%) | 0/5000 mismatches |
| `_cut_rank` full per-cut profile vs. height function, random graphs | 16/200 mismatches | 0/300 mismatches |
| `linear_rank_width` / `linear_rank_width_sa` / `path_clustering_lrw` vs. expected, on cycle/star/complete/path graphs (3 sizes each) | — | 36/36 PASS |
| `tests/test_gf2_rank_and_height.py` | — | 53/53 passed |

Full existing `tests/` suite: same 51 pre-existing failures before and after (in
`test_circuitSolver.py`, `test_generate_graph.py`, `test_tableau.py`), confirmed unrelated via
`git stash`/`git stash pop`. No new failures introduced by this fix.

## Code to paper names mapping

| Code | Paper algorithm | Objective |
|---|---|---|
| `rank_width.py` (`hill_climbing`) | `hill_climbing` | max GF(2) cut-rank (`_cut_rank`) |
| `rank_width_sa.py` (`height_function_sa`) | `height_function_sa` | max stabilizer height function |
| `path_clustering.py` (`path_clustering`) | `path_clustering` | max height function (imports `_full_evaluate`) |
| — | `Minimum_LA_SA` | not examined |
## What else inherits the bug (now fixed, but re-check outputs)

- `rank_width_sa.py::lrw_of_ordering()` — public helper, was unreliable.
- Any direct `gf2_rank` calls in `path_clustering.py` (main cost path uses the height function
  and was fine; direct calls were not).
- `visualise_path_clustering_steps.py` — imports `gf2_rank` for plotting "height" curves;
  previous plots may be wrong and should be regenerated.

## A detail regarding rank_width_sa.py

1. **Everything is imported from `rank_width_sa.py`.** `rank_width_sa.py` now imports
   `_gf2_rank`, `_gf2_rank_bitpacked`, `gf2_rank`, and `_cut_rank` directly from `rank_width.py`
   instead of maintaining its own copy. `path_clustering.py` (which imports `gf2_rank` from
   `rank_width_sa.py`) and `visualise_path_clustering_steps.py` needed no changes — they now
   transparently reference the exact same function objects (`is` identity confirmed). Verified:
   - `rank_width_sa._gf2_rank is rank_width._gf2_rank` (and same for `gf2_rank`, `_cut_rank`) → `True`.
   - Graph-family validation matrix re-run post-refactor: 36/36 PASS (unchanged).
   - `tests/test_gf2_rank_and_height.py`: 53/53 passed (unchanged).
   - `python3 visualise_path_clustering_steps.py` runs end-to-end and regenerates figures with no
     errors (assembled cost 5 → SA-refined 4 emitters on its 30-node demo graph, as before).
   - Full `tests/` suite: same pre-existing failures only (in `test_circuitSolver.py`,
     `test_generate_graph.py`, `test_tableau.py`), no new failures introduced.


