"""
====================
Benchmark all algorithms on every graph in graphs_database_paul/ (QECC to graphs).

For each graph and each algorithm the script computes:
  - ordering found by the algorithm
  - number of emitters
  - number of emitter-CNOTs
  - total number of gates

Algorithms
----------
  lrw          : linear_rank_width          (rank_width.py)
  lrw_sa       : linear_rank_width_sa       (rank_width_sa.py)
  path_clust   : path_clustering_lrw        (path_clustering.py)
  saminla      : run_optimization (SA minLA) (optimization_script.py)

Checkpointing
-------------
Results are stored as individual pickle files under
  /Users/konstantinosrafailrevis/Desktop/PhD/photonic_state_generation/simulations_data/step4_results/<graph_stem>/<algo>.pkl
Each file contains a dict:
  {
    "ordering"  : list,
    "emitters"  : int,
    "cnots"     : int,
    "gates"     : int,
  }
If the file already exists the job is skipped, so you can safely re-run
after a crash / interruption.

Parallelism
-----------
Set NUM_CORES at the top (or override via the environment variable
BENCHMARK_CORES) to control how many worker processes are used.

Quick-test mode
---------------
Set QUICK_TEST = True  by setting this True and choosing a number 
of MAX_VERTICES to only process graphs with vertices up to this number.
It is recomended since some of them require very long time to be computed

"""

from __future__ import annotations

import os
import pickle
import pathlib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional


import networkx as nx
import numpy as np

from optimization_script import run_optimization

# ─────────────────────────────────────────────────────────────────────────────
#  USER SETTINGS  (edit these)
# ─────────────────────────────────────────────────────────────────────────────
# Number of parallel worker processes.  Override with env-var BENCHMARK_CORES.
NUM_CORES: int = int(os.environ.get("BENCHMARK_CORES", 2))

# Set to True to only process graphs with < 20 vertices (quick sanity check).
QUICK_TEST: bool = True
MAX_VERTICES: int = 500 # done: 20, 100, 150, 180
# Where to save results (absolute path).
RESULTS_DIR: str = "/Users/konstantinosrafailrevis/Desktop/PhD/photonic_state_generation/simulations_data/step4_results"

# Path to the graph database.
DB_DIR: str = "graphs_database_paul"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DB_PATH = SCRIPT_DIR / DB_DIR
RESULTS_PATH = pathlib.Path(RESULTS_DIR)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

# --- Define Egde reduction function ---
def edge_reduction_func(input_graph):
    our_graph_original = input_graph.copy()
    # Create a mapping dictionary: (0, 0) -> 0, (0, 1) -> 1, ..., (2, 2) -> 8
    mapping = {node: i for i, node in enumerate(our_graph_original.nodes())}

    # Create a new graph with integer labels
    our_graph = nx.relabel_nodes(our_graph_original, mapping)
    graph , circ, statistics = run_optimization(our_graph, edge_reduction=True, minLA=False, opt_CNOT= False)
    return graph


def _result_path(graph_stem: str, algo: str) -> pathlib.Path:
    return RESULTS_PATH / graph_stem / f"{algo}.pkl"


def _save_result(graph_stem: str, algo: str, result: dict) -> None:
    path = _result_path(graph_stem, algo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)


def _load_result(graph_stem: str, algo: str) -> Optional[dict]:
    path = _result_path(graph_stem, algo)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _relabel_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure nodes are labelled 0..n-1 (required by run_optimization)."""
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping)


def _circuit_stats_from_ordering(graph: nx.Graph, ordering: list) -> tuple[int, int, int]:
    """Given a graph with integer-labelled nodes and an emission ordering,
    compute (emitters, cnots, gates) using the circuit solver.

    The ordering produced by the lrw/SA functions uses the *original* node
    labels of the graph passed to them.  Because we always relabel before
    calling run_optimization we need the ordering expressed as 0..n-1 indices.
    """
    from lib.circuitSolver import algorithm, generationSequence
    from lib.generate_graph import GraphstateGenerator
    from lib.tableau import stabTableau

    adj_matrix = nx.adjacency_matrix(graph, nodelist=ordering).toarray()
    tableau = stabTableau(GraphstateGenerator.get_tableau_from_adj(adj_matrix))
    inv_ops, num_cnots = algorithm(tableau, optimize=True, return_num_cnots=True)
    circuit, num_gates = generationSequence(inv_ops, return_n_unitaries=True)
    num_emitters = circuit.num_qubits - graph.number_of_nodes()
    return num_emitters, num_cnots, num_gates


# ---------------------------------------------------------------------------
#  Per-algorithm workers  (must be top-level functions to be picklable)
# ---------------------------------------------------------------------------

def _run_lrw(graph_stem: str, graph: nx.Graph) -> dict:
    from rank_width import linear_rank_width
    _, ordering = linear_rank_width(graph, seed=42)
    emitters, cnots, gates = _circuit_stats_from_ordering(graph, ordering)
    return {"ordering": ordering, "emitters": emitters, "cnots": cnots, "gates": gates}


def _run_lrw_sa(graph_stem: str, graph: nx.Graph) -> dict:
    from rank_width_sa import linear_rank_width_sa
    _, ordering = linear_rank_width_sa(graph, seed=42)
    emitters, cnots, gates = _circuit_stats_from_ordering(graph, ordering)
    return {"ordering": ordering, "emitters": emitters, "cnots": cnots, "gates": gates}


def _run_path_clust(graph_stem: str, graph: nx.Graph) -> dict:
    from path_clustering import path_clustering_lrw
    _, ordering = path_clustering_lrw(graph, seed=42)
    emitters, cnots, gates = _circuit_stats_from_ordering(graph, ordering)
    return {"ordering": ordering, "emitters": emitters, "cnots": cnots, "gates": gates}


def _run_saminla(graph_stem: str, graph: nx.Graph) -> dict:
    from optimization_script import run_optimization
    _, _, stats = run_optimization(
        graph,
        edge_reduction=False,   # already applied before this call
        minLA=True,
        num_LA=10,
        opt_CNOT=True,
    )
    num_em, num_cnots, num_gates, ordering, _ = stats
    return {"ordering": list(ordering), "emitters": int(num_em),
            "cnots": int(num_cnots), "gates": int(num_gates)}


# Map algo name → worker function
ALGO_WORKERS = {
    "lrw":        _run_lrw,
    "lrw_sa":     _run_lrw_sa,
    "path_clust": _run_path_clust,
    "saminla":    _run_saminla,
}


# ---------------------------------------------------------------------------
#  Job dispatcher  (runs in a worker process)
# ---------------------------------------------------------------------------

def _run_job(job: tuple) -> tuple[str, str, Optional[dict], Optional[str]]:
    """Execute one (graph_stem, algo) job.

    Returns (graph_stem, algo, result_dict_or_None, error_message_or_None).
    """
    graph_stem, algo, graph = job
    worker_fn = ALGO_WORKERS[algo]
    try:
        result = worker_fn(graph_stem, graph)
        return graph_stem, algo, result, None
    except Exception:
        return graph_stem, algo, None, traceback.format_exc()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def build_job_list(
    pkl_files: list[pathlib.Path],
    quick_test: bool,
) -> list[tuple]:
    """Build the list of (graph_stem, algo, graph) tuples that still need running."""
    jobs = []
    skipped = 0
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            graph: nx.Graph = pickle.load(f)

        if quick_test and graph.number_of_nodes() >= MAX_VERTICES:
            continue

        graph_stem = pkl_file.stem

        # Apply edge reduction once, shared across all algorithms
        graph_reduced = edge_reduction_func(graph)

        for algo in ALGO_WORKERS:
            if _load_result(graph_stem, algo) is not None:
                skipped += 1
                continue
            jobs.append((graph_stem, algo, graph_reduced))

    if skipped:
        print(f" {skipped} already-completed job(s) skipped (checkpointing).")
    return jobs


def print_summary(pkl_files: list[pathlib.Path], quick_test: bool) -> None:
    """Print a results table for every completed job."""
    print("\n" + "=" * 90)
    print(f"{'Graph':<45} {'Algo':<12} {'Emitters':>9} {'CNOTs':>8} {'Gates':>8}")
    print("=" * 90)
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            graph: nx.Graph = pickle.load(f)
        if quick_test and graph.number_of_nodes() >= MAX_VERTICES:
            continue
        stem = pkl_file.stem
        for algo in ALGO_WORKERS:
            res = _load_result(stem, algo)
            if res is None:
                print(f"{stem:<45} {algo:<12} {'—':>9} {'—':>8} {'—':>8}")
            else:
                print(f"{stem:<45} {algo:<12} {res['emitters']:>9} "
                      f"{res['cnots']:>8} {res['gates']:>8}")
    print("=" * 90)


def main() -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(DB_PATH.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {DB_PATH}")

    mode_str = f"QUICK TEST (<{MAX_VERTICES} nodes)" if QUICK_TEST else "FULL RUN"
    print(f"\n{'─'*60}")
    print(f"  paul_db_benchmark  |  {mode_str}")
    print(f"  Graphs DB  : {DB_PATH}")
    print(f"  Results    : {RESULTS_PATH}")
    print(f"  Cores      : {NUM_CORES}")
    print(f"  Algorithms : {', '.join(ALGO_WORKERS)}")
    print(f"{'─'*60}\n")

    jobs = build_job_list(pkl_files, QUICK_TEST)

    if not jobs:
        print("Nothing left to compute — all results already exist.")
        print_summary(pkl_files, QUICK_TEST)
        return

    print(f"Submitting {len(jobs)} job(s) across {NUM_CORES} worker(s)...\n")

    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            graph_stem, algo, result, error = future.result()
            if error:
                failed += 1
                print(f"  [ERROR]  {graph_stem}  /  {algo}")
                print("    " + error.replace("\n", "\n    "))
            else:
                _save_result(graph_stem, algo, result)
                completed += 1
                print(
                    f"  [OK]  {graph_stem:<45}  {algo:<12}  "
                    f"emitters={result['emitters']}  "
                    f"cnots={result['cnots']}  "
                    f"gates={result['gates']}"
                )

    print(f"\nDone.  Completed: {completed}  |  Failed: {failed}\n")
    print_summary(pkl_files, QUICK_TEST)


if __name__ == "__main__":
    main()
