"""
====================
Standalone, more robust runtime comparison of SAminLA / Hill Climbing
(rank_width, refined) / Height Function SA (rank_width_sa) / Path Clustering,
replacing the "Runtime for each algorithm" section of
compare_every_algo_version.ipynb.

Why this exists
----------------
The notebook version times each of the 4 algorithms once per graph, always
in the same fixed order (SAminLA, rank_width, rank_width_sa, path_clustering),
on whatever the machine happens to be doing at that moment. That's fragile
for a few reasons:
  - a single sample per graph is very sensitive to a one-off stall
  - a fixed order means any systematic drift (background load ramping up,
    disk cache warming, thermal throttling) always lands on the same
    algorithm first or last, biasing the comparison instead of averaging out
  - there's no record of what else was running on the machine at the time,
    so a weird number later can't be explained, only guessed at

What this script does differently
----------------------------------
  - each graph is timed N_REPEATS times per algorithm, and the per-graph
    result is the median across those reps, not a single sample
  - the algorithm order is reshuffled independently for every repetition,
    so ordering bias cancels out instead of consistently favoring or
    penalizing one algorithm
  - the garbage collector is disabled for the duration of each individual
    timed call, so a GC pause never lands on one algorithm more than another
  - one full untimed warm-up pass runs before any real measurement, so
    first-call overhead (imports, disk cache) doesn't inflate whichever
    algorithm happens to run first
  - the 1-minute load average (os.getloadavg()) is recorded alongside every
    group's measurements, so you can tell after the fact whether a spike was
    the machine being busy with something else rather than a real effect
  - purely sequential, single process, deliberately NOT parallelized -- the
    4 algorithms must never compete with each other (or copies of
    themselves) for CPU, that would reintroduce exactly the kind of noise
    this script exists to remove
  - checkpoints after every (n, p) group, so an interruption never costs
    more than the group in flight

Output
------
`collect` writes runtime_robust_ds{1,2}_{saminla,rw_refined,rw_sa,
path_clustering}.pkl to RESULTS_DIR (same directory the notebook's runtime
files live in, different filenames so nothing there gets overwritten).

`plot` reads those files back and reproduces the same 5 plots the notebook
built (runtime vs p, runtime vs n, each linear and log scale, plus the
residual-vs-path_clustering plot), saved with a "robust_" filename prefix.

Run:
  python runtime.py collect --dataset ds1|ds2|all --reps 3
  python runtime.py plot    --dataset ds1|ds2|all
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import pickle
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from optimization_script import run_optimization
from path_clustering import path_clustering_lrw
from rank_width import linear_rank_width
from rank_width_sa import linear_rank_width_sa

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

N_FIXED = 20
P_FIXED = 0.65

ALGO_NAMES = ["saminla", "rw_refined", "rw_sa", "path_clustering"]
ALGO_LABELS = {
    "saminla":         "Minimum LA SA",
    "rw_refined":      "Hill Climbing",
    "rw_sa":           "Height Function SA",
    "path_clustering": "Path Clustering",
}
ALGO_STYLES = {
    "saminla":         ("o-", "tab:blue"),
    "rw_refined":      ("s-", "tab:orange"),
    "rw_sa":           ("^-", "tab:green"),
    "path_clustering": ("d-", "tab:red"),
}

DEFAULT_REPS = 3
DEFAULT_SEED = 42

# Defaults land at ~3.8h for ds1+ds2 combined at reps=3 (measured from real
# timing), instead of the full dataset's ~15.4h. Pass --graphs-per-group 30
# --n-step 1 to restore full coverage (full resolution, all 30 graphs/group).
#
# n_step subsamples ds2's 25 n-values (6..30) by taking every n_step-th one,
# always including both endpoints -- e.g. n_step=2 keeps n=6,8,10,...,30
# (13 points), n_step=3 keeps n=6,9,12,...,30 (9 points). This trades
# resolution along n for a full-range run, unlike a hard cutoff which would
# drop ds2's largest, most expensive graphs entirely. Has no effect on ds1
# (always n=20, a single value).
DEFAULT_GRAPHS_PER_GROUP = 12
DEFAULT_N_STEP = 2
# ─────────────────────────────────────────────────────────────────────────────


def _ts() -> str:
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


# ---------------------------------------------------------------------------
#  Timing
# ---------------------------------------------------------------------------

def _call_algo(name: str, G_sim: nx.Graph) -> None:
    if name == "saminla":
        run_optimization(G_sim)
    elif name == "rw_refined":
        linear_rank_width(G_sim)
    elif name == "rw_sa":
        linear_rank_width_sa(G_sim)
    elif name == "path_clustering":
        path_clustering_lrw(G_sim)
    else:
        raise ValueError(f"unknown algo {name!r}")


def _timed_call(name: str, G_sim: nx.Graph) -> float:
    """Time one call with GC disabled, so a GC pause can't land unevenly."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter()
        _call_algo(name, G_sim)
        return time.perf_counter() - t0
    finally:
        if was_enabled:
            gc.enable()


def measure_graph(G_sim: nx.Graph, n_reps: int, rng: random.Random) -> dict[str, list[float]]:
    """Time all 4 algorithms n_reps times each on the same (already reduced)
    graph. The order of the 4 algorithms is reshuffled independently for
    every repetition, so systematic drift doesn't consistently favor
    whichever one happens to run first or last."""
    times: dict[str, list[float]] = {name: [] for name in ALGO_NAMES}
    for _ in range(n_reps):
        order = ALGO_NAMES[:]
        rng.shuffle(order)
        for name in order:
            times[name].append(_timed_call(name, G_sim))
    return times


def warm_up(dataset: dict) -> None:
    """One untimed pass over a single graph, so first-call overhead (lazy
    imports, disk cache, etc.) doesn't inflate whichever algorithm happens
    to be measured first for real."""
    first_group = next(iter(dataset.values()))
    G_sim = edge_reduction_func(first_group[0])
    for name in ALGO_NAMES:
        _call_algo(name, G_sim)


# ---------------------------------------------------------------------------
#  Collect
# ---------------------------------------------------------------------------

def _subsample_by_step(values: list[int], step: int) -> set[int]:
    """Every step-th value from a sorted list, always including the last one."""
    if step <= 1:
        return set(values)
    selected = list(values[::step])
    if values[-1] not in selected:
        selected.append(values[-1])
    return set(selected)


def simulate_dataset(
    ds_name: str,
    n_reps: int,
    seed: int,
    graphs_per_group: int | None = None,
    n_step: int = 1,
) -> None:
    dataset = load_dataset(DATASET_FILES[ds_name])
    rng = random.Random(seed)

    print(f"[{_ts()}] Warm-up pass (untimed)...")
    warm_up(dataset)

    groups = sorted(dataset.items(), key=lambda x: x[0][1])
    if n_step > 1:
        all_ns = sorted(set(n for (n, _p) in dataset))
        keep_ns = _subsample_by_step(all_ns, n_step)
        groups = [((n, p), graphs) for (n, p), graphs in groups if n in keep_ns]
    if graphs_per_group is not None:
        groups = [((n, p), graphs[:graphs_per_group]) for (n, p), graphs in groups]
    total_groups = len(groups)
    result_dicts: dict[str, dict] = {name: {} for name in ALGO_NAMES}
    combo_start = time.time()

    print(f"\n{'─'*60}")
    print(f"  [{_ts()}] runtime (robust) — {ds_name}  ({n_reps} rep(s)/graph, median taken)")
    if graphs_per_group is not None or n_step > 1:
        print(f"  scope: graphs_per_group={graphs_per_group or 'all'}, n_step={n_step}"
              f"  (reduced from full dataset for time budget)")
    print(f"  {total_groups} (n,p) group(s) to process")
    print(f"{'─'*60}")

    for group_idx, ((n, p), graphs) in enumerate(groups, start=1):
        group_start = time.time()
        load1, load5, load15 = os.getloadavg()
        print(f"[{_ts()}] group {group_idx}/{total_groups}  n={n}, p={p}  "
              f"({len(graphs)} graphs)  load avg={load1:.2f}/{load5:.2f}/{load15:.2f} ... ",
              end="", flush=True)

        raw: dict[str, list[float]] = {name: [] for name in ALGO_NAMES}
        for G in graphs:
            G_sim = edge_reduction_func(G)
            per_algo_reps = measure_graph(G_sim, n_reps, rng)
            for name in ALGO_NAMES:
                raw[name].append(float(np.median(per_algo_reps[name])))

        for name in ALGO_NAMES:
            arr = np.array(raw[name])
            result_dicts[name][(n, p)] = {
                "time_mean":     arr.mean(),
                "time_std":      arr.std(),
                "time_raw":      arr,
                "load_avg_1min": load1,
            }

        group_elapsed = time.time() - group_start
        combo_elapsed = time.time() - combo_start
        avg_per_group = combo_elapsed / group_idx
        remaining = avg_per_group * (total_groups - group_idx)
        print(f"done in {_fmt_secs(group_elapsed)}  "
              f"(elapsed {_fmt_secs(combo_elapsed)}, ~{_fmt_secs(remaining)} left)")
        for name in ALGO_NAMES:
            d = result_dicts[name][(n, p)]
            print(f"    {ALGO_LABELS[name]:>18s}: {d['time_mean']:.3f} ± {d['time_std']:.3f} s")

        # Checkpoint after every (n, p) group, not just at the very end.
        _save_result_dicts(ds_name, result_dicts)

    print(f"[{_ts()}] Done with {ds_name}  "
          f"({total_groups}/{total_groups} groups, total {_fmt_secs(time.time() - combo_start)}).")


def _save_result_dicts(ds_name: str, result_dicts: dict[str, dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, res_dict in result_dicts.items():
        fpath = RESULTS_DIR / f"runtime_robust_{ds_name}_{name}.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(res_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [checkpoint saved -> {RESULTS_DIR}/runtime_robust_{ds_name}_*.pkl]", flush=True)


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------

def _load_robust_results(ds_name: str) -> dict[str, dict]:
    out = {}
    for name in ALGO_NAMES:
        fpath = RESULTS_DIR / f"runtime_robust_{ds_name}_{name}.pkl"
        with open(fpath, "rb") as f:
            out[name] = pickle.load(f)
    return out


def _extract_runtime(res_dict: dict, keys: list) -> tuple[np.ndarray, np.ndarray]:
    means = [res_dict[k]["time_mean"] if k in res_dict else float("nan") for k in keys]
    stds = [res_dict[k]["time_std"] if k in res_dict else float("nan") for k in keys]
    return np.array(means), np.array(stds)


def _common_keys(results: dict[str, dict], fixed_idx: int, fixed_val) -> list:
    """Intersection of (n,p) keys present in every algorithm's results, for
    the given fixed axis (n=N_FIXED for ds1, p=P_FIXED for ds2)."""
    all_sets = [
        set(k for k in res_dict if k[fixed_idx] == fixed_val)
        for res_dict in results.values()
    ]
    return sorted(set.intersection(*all_sets)) if all_sets else []


def plot_runtime_vs_x(
    ds_name: str,
    results: dict[str, dict],
    x_values: list,
    keys: list,
    x_label: str,
    title: str,
    out_prefix: str,
) -> None:
    for log_y, log_tag, ylabel_suffix in [(False, "", ""), (True, "_log", "  [log scale]")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in ALGO_NAMES:
            marker, color = ALGO_STYLES[name]
            means, stds = _extract_runtime(results[name], keys)
            ax.errorbar(x_values, means, yerr=stds, fmt=marker, color=color,
                        label=ALGO_LABELS[name], capsize=3, markersize=5, linewidth=1.5)
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(f"Runtime (s){ylabel_suffix}", fontsize=13)
        ax.set_title(f"{title} (robust, median of reps)", fontsize=14)
        if log_y:
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        fpath = RESULTS_DIR / f"robust_{out_prefix}{log_tag}.pdf"
        plt.savefig(fpath, dpi=150)
        plt.show()
        print(f"Saved -> {fpath}")


def plot_residual_vs_path_clustering(
    ds_label: str,
    x_values: list,
    x_label: str,
    keys: list,
    results: dict[str, dict],
    out_name: str,
) -> None:
    ref_m, ref_s = _extract_runtime(results["path_clustering"], keys)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Runtime residual vs Path Clustering (robust)  —  {ds_label}\n"
        r"$({\rm algo} - {\rm Path\ Clustering})\times 100\;/\;{\rm Path\ Clustering}$  [%]",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0, color="tab:red", lw=1.5, ls="--", label="Path Clustering (ref)")

    for name in ["saminla", "rw_refined", "rw_sa"]:
        marker, color = ALGO_STYLES[name]
        algo_m, algo_s = _extract_runtime(results[name], keys)
        residual_m = (algo_m - ref_m) * 100.0 / ref_m
        residual_s = (100.0 / ref_m) * np.sqrt(algo_s**2 + (algo_m / ref_m)**2 * ref_s**2)
        ax.errorbar(x_values, residual_m, yerr=residual_s, fmt=marker, color=color,
                    capsize=3, lw=2, ms=5, label=ALGO_LABELS[name])

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Relative diff. [%]", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    fpath = RESULTS_DIR / f"robust_{out_name}.pdf"
    plt.savefig(fpath, dpi=150)
    plt.show()
    print(f"Saved -> {fpath}")


def make_plots(ds_name: str) -> None:
    results = _load_robust_results(ds_name)

    if ds_name == "ds1":
        keys = _common_keys(results, fixed_idx=0, fixed_val=N_FIXED)
        x_values = [p for (_n, p) in keys]
        plot_runtime_vs_x(
            ds_name, results, x_values, keys,
            x_label="Edge probability p",
            title=f"Runtime vs graph density (n = {N_FIXED}, ER graphs)",
            out_prefix="runtime_vs_density_ds1",
        )
        plot_residual_vs_path_clustering(
            f"Dataset 1  (n = {N_FIXED}, varying p)", x_values, "Edge probability p",
            keys, results, "runtime_residual_vs_path_clustering_ds1",
        )
    else:
        keys = _common_keys(results, fixed_idx=1, fixed_val=P_FIXED)
        x_values = [n for (n, _p) in keys]
        plot_runtime_vs_x(
            ds_name, results, x_values, keys,
            x_label="Photon number n",
            title=f"Runtime vs graph size (p = {P_FIXED}, ER graphs)",
            out_prefix="runtime_vs_nodes_ds2",
        )
        plot_residual_vs_path_clustering(
            f"Dataset 2  (p = {P_FIXED}, varying n)", x_values, "Photon number n",
            keys, results, "runtime_residual_vs_path_clustering_ds2",
        )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["collect", "plot"], help="Collect timing data, or plot previously collected data.")
    parser.add_argument("--dataset", choices=["ds1", "ds2", "all"], default="all",
                        help="Which dataset to run/plot (default: all).")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help=f"Repetitions per graph per algorithm, median is used (default: {DEFAULT_REPS}). "
                             "collect only.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Seed for the per-repetition algorithm-order shuffle (default: {DEFAULT_SEED}). "
                             "collect only.")
    parser.add_argument("--graphs-per-group", type=int, default=DEFAULT_GRAPHS_PER_GROUP,
                        help=f"Graphs sampled per (n,p) group, out of 30 available (default: "
                             f"{DEFAULT_GRAPHS_PER_GROUP}). Pass 30 for the full dataset. collect only.")
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP,
                        help=f"Subsample ds2's n-values by taking every n-th one, both endpoints "
                             f"always included (default: {DEFAULT_N_STEP}). Pass 1 for full n-resolution. "
                             "Has no effect on ds1 (always n=20). collect only.")
    args = parser.parse_args()

    datasets = ["ds1", "ds2"] if args.dataset == "all" else [args.dataset]

    if args.mode == "collect":
        for ds_name in datasets:
            simulate_dataset(ds_name, args.reps, args.seed,
                              graphs_per_group=args.graphs_per_group, n_step=args.n_step)
    else:
        for ds_name in datasets:
            make_plots(ds_name)


if __name__ == "__main__":
    main()
