"""
====================
Generic element-by-element comparison of two datasets (e.g. a buggy vs a
bug-fixed algorithm run), matched by graph key.

Each dataset is described by a small spec dict of one of two kinds:

  kind="pkl"  — per-graph checkpoint directory, as produced by
                rhg_runs.py / runs_shor_full.py:
                  <results_dir>/<graph_stem>/<algo>.pkl
                each pickle is a dict with the raw metric keys.

  kind="csv"  — a single CSV file, as produced by the batch_process_*.ipynb
                notebooks (e.g. simulations_data/final_algo_comparison/*.csv),
                keyed by a column (default "path") with one column per metric.

For every graph present in BOTH datasets, this prints (per metric):
  - mean(A), mean(B), mean difference (B - A)
  - paired t-test p-value
  - Wilcoxon signed-rank p-value
and plots the mean paired % difference ((B - A) / A * 100) with a 95%
confidence interval, saved to PLOT_OUT.

Presets are registered in PRESETS below. Pick one on the command line:

  python compare_fix_bug.py path_clustering
  python compare_fix_bug.py hill_climbing
  python compare_fix_bug.py shor
  python compare_fix_bug.py rhg
  python compare_fix_bug.py qecc
  python compare_fix_bug.py            # lists available presets

To add a new comparison, add another entry to PRESETS. Each preset compares
the full METRICS set by default; a preset can override this with its own
"metrics" list if its data doesn't have all of them (e.g. qecc_runs.py never
recorded runtime, so the "qecc" preset only compares emitters/cnots/gates).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import pickle
import sys
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
#  USER SETTINGS  (edit these)
# ─────────────────────────────────────────────────────────────────────────────
# Canonical metric names used throughout this script, and their display labels.
METRICS = ["emitters", "cnots", "gates", "runtime"]
METRIC_LABELS = {
    "emitters": "Emitters",
    "cnots":    "CNOTs",
    "gates":    "Gates",
    "runtime":  "Runtime (s)",
}

FINAL_ALGO_COMPARISON_DIR = pathlib.Path(
    "/Users/konstantinosrafailrevis/Desktop/PhD/photonic_state_generation"
    "/simulations_data/final_algo_comparison"
)
SHOR_RESULTS_DIR = pathlib.Path(
    "/Users/konstantinosrafailrevis/Desktop/PhD/photonic_state_generation"
    "/simulations_data/step4_results_shor_red"
)
RHG_RESULTS_DIR = pathlib.Path(
    "/Users/konstantinosrafailrevis/Desktop/PhD/photonic_state_generation"
    "/simulations_data/step4_results_rhg"
)
QECC_RESULTS_DIR = pathlib.Path(
    "/Users/konstantinosrafailrevis/Desktop/PhD/photonic_state_generation"
    "/simulations_data/step4_results"
)

# ── Preset registry ─────────────────────────────────────────────────────────
# Each preset maps a name -> (label_a, dataset_a, label_b, dataset_b).
# Add new comparisons here.
PRESETS: dict[str, dict[str, Any]] = {
    "path_clustering": dict(
        label_a="path_clustering (buggy)",
        dataset_a=dict(
            kind="csv",
            path=FINAL_ALGO_COMPARISON_DIR / "graph_stats_path_clustering.csv",
            key_col="path",
            metrics={
                "emitters": "num_emitters_path",
                "cnots":    "num_emitter_CNOTs_path",
                "gates":    "num_gates_path",
                "runtime":  "run_time_path",
            },
        ),
        label_b="path_clustering (fixed)",
        dataset_b=dict(
            kind="csv",
            path=FINAL_ALGO_COMPARISON_DIR / "graph_stats_path_clustering_fix_bug.csv",
            key_col="path",
            metrics={
                "emitters": "num_emitters_path",
                "cnots":    "num_emitter_CNOTs_path",
                "gates":    "num_gates_path",
                "runtime":  "run_time_path",
            },
        ),
    ),
    "hill_climbing": dict(
        label_a="hill_climbing (buggy)",
        dataset_a=dict(
            kind="csv",
            path=FINAL_ALGO_COMPARISON_DIR / "graph_stats_rank_width.csv",
            key_col="path",
            metrics={
                "emitters": "num_emitters_rw",
                "cnots":    "num_emitter_CNOTs_rw",
                "gates":    "num_gates_rw",
                "runtime":  "run_time_rw",
            },
        ),
        label_b="hill_climbing (fixed)",
        dataset_b=dict(
            kind="csv",
            path=FINAL_ALGO_COMPARISON_DIR / "graph_stats_rank_width_fixed_bug.csv",
            key_col="path",
            metrics={
                "emitters": "num_emitters_rw",
                "cnots":    "num_emitter_CNOTs_rw",
                "gates":    "num_gates_rw",
                "runtime":  "run_time_rw",
            },
        ),
    ),
    "shor": dict(
        label_a="hill_climbing (buggy)",
        dataset_a=dict(
            kind="pkl",
            results_dir=SHOR_RESULTS_DIR,
            algo="lrw",
            metrics={"emitters": "emitters", "cnots": "cnots", "gates": "gates", "runtime": "ordering_time_s"},
        ),
        label_b="hill_climbing_fix_bug (fixed)",
        dataset_b=dict(
            kind="pkl",
            results_dir=SHOR_RESULTS_DIR,
            algo="lrw_fix_bug",
            metrics={"emitters": "emitters", "cnots": "cnots", "gates": "gates", "runtime": "ordering_time_s"},
        ),
    ),
    # RHG lattices (rhg_runs.py). RGH_4_4_4's "lrw_fix_bug" run never finished,
    # so it has no lrw_fix_bug.pkl — load_pkl_dataset only pairs graph_stems
    # present in BOTH datasets, so it's automatically excluded (comparison
    # covers everything up to and including (3,4,4)).
    "rhg": dict(
        label_a="hill_climbing (buggy)",
        dataset_a=dict(
            kind="pkl",
            results_dir=RHG_RESULTS_DIR,
            algo="lrw",
            metrics={"emitters": "emitters", "cnots": "cnots", "gates": "gates", "runtime": "runtime_s"},
        ),
        label_b="hill_climbing (fixed)",
        dataset_b=dict(
            kind="pkl",
            results_dir=RHG_RESULTS_DIR,
            algo="lrw_fix_bug",
            metrics={"emitters": "emitters", "cnots": "cnots", "gates": "gates", "runtime": "runtime_s"},
        ),
    ),
    # QECC families (qecc_runs.py). Its pkl files only ever recorded
    # ordering/emitters/cnots/gates -- no runtime -- so this preset overrides
    # "metrics" to drop "runtime" from the comparison entirely.
    "qecc": dict(
        label_a="hill_climbing (buggy)",
        dataset_a=dict(
            kind="pkl",
            results_dir=QECC_RESULTS_DIR,
            algo="lrw",
            metrics={"emitters": "emitters", "cnots": "cnots", "gates": "gates"},
        ),
        label_b="hill_climbing (fixed)",
        dataset_b=dict(
            kind="pkl",
            results_dir=QECC_RESULTS_DIR,
            algo="lrw_fix_bug",
            metrics={"emitters": "emitters", "cnots": "cnots", "gates": "gates"},
        ),
        metrics=["emitters", "cnots", "gates"],
    ),
}

CI_LEVEL = 0.95
# ─────────────────────────────────────────────────────────────────────────────


def load_pkl_dataset(spec: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Load a per-graph checkpoint directory: <results_dir>/<graph_stem>/<algo>.pkl."""
    results_dir: pathlib.Path = spec["results_dir"]
    algo: str = spec["algo"]
    metric_map: dict[str, str] = spec["metrics"]

    data: dict[str, dict[str, float]] = {}
    for stem_dir in sorted(results_dir.iterdir()):
        if not stem_dir.is_dir():
            continue
        pkl_path = stem_dir / f"{algo}.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            res = pickle.load(f)
        vals = {m: res.get(k) for m, k in metric_map.items()}
        if any(v is None for v in vals.values()):
            continue
        data[stem_dir.name] = vals
    return data


def load_csv_dataset(spec: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Load a CSV file, keyed by spec['key_col'], with one column per metric."""
    csv_path: pathlib.Path = spec["path"]
    key_col: str = spec.get("key_col", "path")
    metric_map: dict[str, str] = spec["metrics"]

    data: dict[str, dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get(key_col)
            if not key:
                continue
            vals: dict[str, float] = {}
            ok = True
            for m, col in metric_map.items():
                raw = row.get(col)
                try:
                    vals[m] = float(raw)
                except (TypeError, ValueError):
                    ok = False
                    break
            if ok:
                data[key] = vals
    return data


def load_dataset(spec: dict[str, Any]) -> dict[str, dict[str, float]]:
    kind = spec["kind"]
    if kind == "pkl":
        return load_pkl_dataset(spec)
    if kind == "csv":
        return load_csv_dataset(spec)
    raise ValueError(f"Unknown dataset kind: {kind!r}")


def collect_pairs(
    data_a: dict[str, dict[str, float]],
    data_b: dict[str, dict[str, float]],
    metrics: list[str],
) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]]]:
    """Element-by-element pairing: every key present (with all metrics) in BOTH datasets."""
    keys = sorted(set(data_a) & set(data_b))

    vals_a: dict[str, list[float]] = {m: [] for m in metrics}
    vals_b: dict[str, list[float]] = {m: [] for m in metrics}
    used_keys: list[str] = []

    for k in keys:
        used_keys.append(k)
        for m in metrics:
            vals_a[m].append(data_a[k][m])
            vals_b[m].append(data_b[k][m])

    return used_keys, vals_a, vals_b


def _mean_and_ci(diffs: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    """Return (mean, half-width) of a (1-level) t-distribution confidence interval."""
    n = len(diffs)
    mean = diffs.mean()
    if n < 2:
        return mean, 0.0
    sem = diffs.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - (1 - level) / 2, df=n - 1)
    return mean, t_crit * sem


def plot_differences(
    vals_a: dict[str, list[float]],
    vals_b: dict[str, list[float]],
    label_a: str,
    label_b: str,
    out_path: pathlib.Path,
    metrics: list[str],
) -> None:
    """Bar chart of the mean paired % difference ((B - A) / A * 100) per metric, with CI whiskers."""
    n = len(next(iter(vals_a.values())))

    means, cis, n_used = [], [], []
    for m in metrics:
        a = np.asarray(vals_a[m], dtype=float)
        b = np.asarray(vals_b[m], dtype=float)
        valid = a != 0  # exclude pairs with a zero denominator (undefined % change)
        pct_diffs = (b[valid] - a[valid]) / a[valid] * 100.0
        mean, ci = _mean_and_ci(pct_diffs)
        means.append(mean)
        cis.append(ci)
        n_used.append(int(valid.sum()))

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.5), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, m, mean, ci, n_m in zip(axes, metrics, means, cis, n_used):
        color = "#4CAF50" if mean <= 0 else "#F44336"
        ax.bar([0], [mean], yerr=[ci], capsize=8, color=color,
               edgecolor="black", linewidth=0.8, width=0.5,
               error_kw=dict(elinewidth=1.4, capthick=1.4))
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks([0])
        excl = f"\n(n={n_m}, {n - n_m} excl. zero-denom)" if n_m != n else ""
        ax.set_xticklabels([METRIC_LABELS.get(m, m) + excl], fontsize=8)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylabel("Mean % difference (B − A) / A × 100")
        ax.set_title(METRIC_LABELS.get(m, m), fontweight="bold", pad=22)
        ax.margins(y=0.25)  # headroom so the annotation below never touches the title

        tip_y = mean + ci if mean >= 0 else mean - ci
        ax.annotate(
            f"{mean:+.1f}%",
            xy=(0, tip_y),
            xytext=(0, 6 if mean >= 0 else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=10,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"{label_b} vs {label_a} — Mean % Paired Difference "
        f"({int(CI_LEVEL * 100)}% CI, n={n} graphs)",
        fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot -> {out_path}")
    plt.show()


def run_preset(name: str) -> None:
    preset = PRESETS[name]
    label_a = preset["label_a"]
    label_b = preset["label_b"]
    dataset_a = preset["dataset_a"]
    dataset_b = preset["dataset_b"]
    metrics = preset.get("metrics", METRICS)
    plot_out = pathlib.Path(f"compare_fix_bug_diff_{name}.png")

    data_a = load_dataset(dataset_a)
    data_b = load_dataset(dataset_b)
    print(f"Dataset A ({label_a}): {len(data_a)} graph(s) loaded.")
    print(f"Dataset B ({label_b}): {len(data_b)} graph(s) loaded.")

    graph_keys, vals_a, vals_b = collect_pairs(data_a, data_b, metrics)
    n = len(graph_keys)

    if n == 0:
        print("No graphs found in both datasets — nothing to compare.")
        return

    print(f"\nCompared {n} graph(s) present in both datasets.\n")

    header = (
        f"{'Metric':18s}{'mean A':>14s}{'mean B':>14s}"
        f"{'mean Δ':>12s}{'Wilcoxon p':>14s}{'paired t p':>14s}"
    )
    print(header)
    print("-" * len(header))

    for m in metrics:
        a = vals_a[m]
        b = vals_b[m]
        diff = [bi - ai for ai, bi in zip(a, b)]
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        mean_diff = sum(diff) / n

        if any(d != 0 for d in diff):
            w_p = stats.wilcoxon(a, b).pvalue
        else:
            w_p = float("nan")
        t_p = stats.ttest_rel(a, b).pvalue

        print(f"{m:18s}{mean_a:>14.3f}{mean_b:>14.3f}{mean_diff:>+12.3f}{w_p:>14.4g}{t_p:>14.4g}")

    n_changed = sum(
        1 for i in range(n)
        if any(vals_a[m][i] != vals_b[m][i] for m in metrics)
    )

    print(f"\n  p < 0.05 on either test  =>  statistically significant change from A to B.")
    print(f"  mean Δ = mean(B) - mean(A); negative = B uses fewer resources / less time than A.")
    print(f"\n  {n_changed}/{n} graph(s) have at least one metric that differs between A and B.")

    plot_differences(vals_a, vals_b, label_a, label_b, plot_out, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "preset",
        nargs="?",
        choices=sorted(PRESETS),
        help="Which comparison to run.",
    )
    args = parser.parse_args()

    if args.preset is None:
        print("Available presets:")
        for name in sorted(PRESETS):
            p = PRESETS[name]
            print(f"  {name:18s} {p['label_a']}  vs  {p['label_b']}")
        print("\nRun: python compare_fix_bug.py <preset>")
        sys.exit(0)

    run_preset(args.preset)


if __name__ == "__main__":
    main()
