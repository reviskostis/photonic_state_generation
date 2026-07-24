"""
====================
Standalone re-run of everything the "rank_width" algorithm did in
compare_every_algo_version.ipynb (cells under "Emission with rank_width and
[NO] edge reduction (all 4 ordering variants)"), for the bug-fixed
rank_width.py.

For every graph, rank_width produces 4 candidate orderings:
  spectral, rcm, mindegree   (Phase A/B only, refine=False)
  refined                    (Phase A/B + C hill-climb, refine=True)
and each is evaluated independently through run_optimization to get
(emitters, cnots, gates). This is done for both datasets (ds1: fixed n=20,
varying p; ds2: fixed p=0.65, varying n) and both with and without edge
reduction applied first -- 2 datasets x 2 edge-reduction modes x 4 ordering
variants = 16 result sets, matching the notebook's 16 saved .pkl files:

  results_ds{1,2}_rw_{spectral,rcm,mindegree,refined}_edge_red[_no].pkl

This script writes the same 16 files with a "_fix_bug" suffix appended
(e.g. results_ds1_rw_refined_edge_red_fix_bug.pkl), so the original
pre-fix results on disk are never touched -- you can load both and compare.

Checkpointing
-------------
The notebook only saved each result set once, after its (n,p)/(n) loop
fully finished. This script re-saves after every (n,p) key instead, so an
interruption partway through never loses more than the group in flight.
Each save fully overwrites the accumulated-so-far dict for that ordering
variant, so it is always safe to just re-run this script from scratch.

Parallelism
-----------
Graphs within a (n, p) group are independent, so they're farmed out to a
ProcessPoolExecutor (one worker pool per dataset x edge-reduction combo,
reused across all of that combo's groups). A single graph's failure is
caught in the worker and printed, not fatal to the whole run. Set
NUM_CORES below (or override via the environment variable
BENCHMARK_CORES) to control how many worker processes are used.

Run: python compare_rank_width_fix_bug.py [--dataset ds1|ds2|all] [--edge-reduction with|without|all]
     (both default to "all")
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import networkx as nx
import numpy as np

from optimization_script import run_optimization
from rank_width import _relabel_to_int, generate_initial_orderings, linear_rank_width

# ─────────────────────────────────────────────────────────────────────────────
#  USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR / "step_2_dataset_small"
RESULTS_DIR = pathlib.Path(
    "/Users/konstantinosrafailrevis/Desktop/PhD/photonic_state_generation/simulations_data/step2_results_small"
)

DATASET_FILES = {
    "ds1": "dataset_fixed_n20_vary_p.pkl",
    "ds2": "dataset_fixed_p065_vary_n.pkl",
}

ORDERING_NAMES = ["spectral", "rcm", "mindegree", "refined"]

# Number of parallel worker processes. Override with env-var BENCHMARK_CORES.
# NOTE: default is deliberately conservative (2), not os.cpu_count() -- this
# machine is frequently running other simulations on most cores already, and
# requesting more workers than are actually free just causes contention
# (measured ~2.3x speedup with 8 workers against ~3 free cores, no better
# than requesting fewer workers). Set BENCHMARK_CORES explicitly to match
# however many cores are actually free at run time.
NUM_CORES: int = int(os.environ.get("BENCHMARK_CORES", 4))
# ─────────────────────────────────────────────────────────────────────────────


def _ts() -> str:
    """Current wall-clock time, HH:MM:SS, for prefixing progress lines."""
    return time.strftime("%H:%M:%S")


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.1f}s"
    return f"{s/60:.1f}min"


def edge_reduction_func(input_graph: nx.Graph) -> nx.Graph:
    mapping = {node: i for i, node in enumerate(input_graph.nodes())}
    relabeled = nx.relabel_nodes(input_graph.copy(), mapping)
    graph, _circ, _stats = run_optimization(relabeled, edge_reduction=True, minLA=False, opt_CNOT=False)
    return graph


def load_dataset(filename: str) -> dict:
    with open(DATASET_DIR / filename, "rb") as f:
        return pickle.load(f)


def rank_width_variants_for_graph(G: nx.Graph, apply_edge_reduction: bool) -> dict[str, tuple[int, int, int]]:
    """Return {ordering_name: (emitters, cnots, gates)} for one graph."""
    G_sim = edge_reduction_func(G) if apply_edge_reduction else G

    G_int, _old_to_new, new_to_old = _relabel_to_int(G_sim)
    candidates = generate_initial_orderings(G_int)  # [spectral, rcm, mindegree]

    out: dict[str, tuple[int, int, int]] = {}
    for name, ordering_int in zip(ORDERING_NAMES[:3], candidates):
        original_ordering = [new_to_old[v] for v in ordering_int]
        ordered_graph = nx.relabel_nodes(G_sim, {old: new for new, old in enumerate(original_ordering)})
        _, _, stats = run_optimization(ordered_graph, edge_reduction=False, minLA=False, opt_CNOT=True)
        out[name] = (stats[0], stats[1], stats[2])

    _, refined_ordering = linear_rank_width(G_sim, refine=True)
    ordered_graph = nx.relabel_nodes(G_sim, {old: new for new, old in enumerate(refined_ordering)})
    _, _, stats = run_optimization(ordered_graph, edge_reduction=False, minLA=False, opt_CNOT=True)
    out["refined"] = (stats[0], stats[1], stats[2])

    return out


def _graph_job(job: tuple) -> tuple[Optional[dict], Optional[str]]:
    """Worker: compute all 4 ordering variants for one graph.

    Must be a top-level function (picklable) to run in a ProcessPoolExecutor.
    Returns (variants_dict_or_None, error_message_or_None).
    """
    G, apply_edge_reduction = job
    try:
        return rank_width_variants_for_graph(G, apply_edge_reduction), None
    except Exception:
        return None, traceback.format_exc()


def simulate_dataset(
    ds_name: str,
    apply_edge_reduction: bool,
    combo_idx: int = 1,
    combo_total: int = 1,
) -> None:
    dataset = load_dataset(DATASET_FILES[ds_name])
    suffix = "_edge_red" if apply_edge_reduction else "_edge_red_no"
    mode_str = "WITH edge reduction" if apply_edge_reduction else "WITHOUT edge reduction"

    groups = sorted(dataset.items(), key=lambda x: x[0][1])
    total_groups = len(groups)
    combo_start = time.time()  # local timing only, never written to disk

    print(f"\n{'─'*60}")
    print(f"  [{_ts()}] rank_width — {ds_name} — {mode_str}  "
          f"({NUM_CORES} worker(s))  [combo {combo_idx}/{combo_total}]")
    print(f"  {total_groups} (n,p) group(s) to process")
    print(f"{'─'*60}")

    result_dicts: dict[str, dict] = {name: {} for name in ORDERING_NAMES}

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        for group_idx, ((n, p), graphs) in enumerate(groups, start=1):
            raw = {name: {"emitters": [], "cnots": [], "gates": []} for name in ORDERING_NAMES}

            group_start = time.time()
            print(f"[{_ts()}] group {group_idx}/{total_groups}  n={n}, p={p}  "
                  f"({len(graphs)} graphs) ... ", end="", flush=True)

            jobs = [(G, apply_edge_reduction) for G in graphs]
            n_failed = 0
            for variants, error in executor.map(_graph_job, jobs):
                if error:
                    n_failed += 1
                    print(f"\n    ERROR on one graph:\n    " + error.replace("\n", "\n    "))
                    continue
                for name, (emitters, cnots, gates) in variants.items():
                    raw[name]["emitters"].append(emitters)
                    raw[name]["cnots"].append(cnots)
                    raw[name]["gates"].append(gates)

            if n_failed:
                print(f"    ({n_failed}/{len(graphs)} graph(s) failed and were skipped)")

            for name in ORDERING_NAMES:
                em = np.array(raw[name]["emitters"])
                cn = np.array(raw[name]["cnots"])
                gt = np.array(raw[name]["gates"])
                result_dicts[name][(n, p)] = {
                    "emitters_mean": em.mean(), "emitters_std": em.std(), "emitters_raw": em,
                    "cnots_mean":    cn.mean(), "cnots_std":    cn.std(), "cnots_raw":    cn,
                    "gates_mean":    gt.mean(), "gates_std":    gt.std(), "gates_raw":    gt,
                }

            for name in ORDERING_NAMES:
                d = result_dicts[name][(n, p)]
                print(f"\n    {name:>10s}: emitters={d['emitters_mean']:.1f}±{d['emitters_std']:.1f}  "
                      f"cnots={d['cnots_mean']:.1f}±{d['cnots_std']:.1f}  "
                      f"gates={d['gates_mean']:.1f}±{d['gates_std']:.1f}", end="")
            print()

            # Purely informational timing, printed only, never stored.
            group_elapsed = time.time() - group_start
            combo_elapsed = time.time() - combo_start
            avg_per_group = combo_elapsed / group_idx
            remaining = avg_per_group * (total_groups - group_idx)
            print(f"[{_ts()}] group {group_idx}/{total_groups} done in {_fmt_secs(group_elapsed)}  "
                  f"(combo elapsed {_fmt_secs(combo_elapsed)}, "
                  f"~{_fmt_secs(remaining)} left in this combo)")

            # Checkpoint after every (n, p) group, not just at the very end.
            # Same result_dicts / same fields as before, unchanged by the timing additions.
            _save_result_dicts(ds_name, suffix, result_dicts)

    print(f"[{_ts()}] Done with {ds_name} {mode_str}  "
          f"({total_groups}/{total_groups} groups, total {_fmt_secs(time.time() - combo_start)}).")


def _save_result_dicts(ds_name: str, suffix: str, result_dicts: dict[str, dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, res_dict in result_dicts.items():
        fpath = RESULTS_DIR / f"results_{ds_name}_rw_{name}{suffix}_fix_bug.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(res_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [checkpoint saved -> {RESULTS_DIR}/results_{ds_name}_rw_*{suffix}_fix_bug.pkl]", flush=True)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ds1", "ds2", "all"], default="all",
                        help="Which dataset to re-run rank_width on (default: all).")
    parser.add_argument("--edge-reduction", choices=["with", "without", "all"], default="all",
                        help="Whether to run with edge reduction, without, or both (default: all).")
    args = parser.parse_args()

    datasets = ["ds1", "ds2"] if args.dataset == "all" else [args.dataset]
    edge_modes: list[bool] = []
    if args.edge_reduction in ("with", "all"):
        edge_modes.append(True)
    if args.edge_reduction in ("without", "all"):
        edge_modes.append(False)

    print(f"[{_ts()}] Cores: {NUM_CORES} (override with BENCHMARK_CORES env var)")

    combos = [(ds_name, mode) for ds_name in datasets for mode in edge_modes]
    for combo_idx, (ds_name, apply_edge_reduction) in enumerate(combos, start=1):
        simulate_dataset(ds_name, apply_edge_reduction, combo_idx=combo_idx, combo_total=len(combos))


if __name__ == "__main__":
    main()
