"""
Save Shor algorithm MBQC graph states for all (N, a) configurations.

For each (N, a) pair where a is coprime to N and N in [3..17]:
  - The pattern is reduced (standardize, shift signals, Pauli measurements).
  - Only the **largest connected component** of the reduced graph is saved.

Each graph is saved as an individual pickle file containing a plain
nx.Graph, compatible with the step4_runs pipeline:

    with open(path, "rb") as f:
        graph = pickle.load(f)   # -> nx.Graph

Output structure:
  shor_graph_database/
    full/
      shor_N3_a2.pkl                      # largest component for N=3, a=2
      ...
"""

from __future__ import annotations

import os
import time
import signal
import pickle

import random

import numpy as np
import networkx as nx
from math import gcd, log2, ceil

from graphix import Circuit


# ── Import circuit-building helpers from the notebook's logic ──────────

def controlled_phase(circuit, control, target, angle):
    circuit.rz(target, angle / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -angle / 2)
    circuit.cnot(control, target)
    circuit.rz(control, angle / 2)


def inverse_qft(circuit, qubits):
    n = len(qubits)
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            k = j - i + 1
            angle = -2 * np.pi / (2 ** k)
            controlled_phase(circuit, qubits[j], qubits[i], angle)
        circuit.h(qubits[i])
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - 1 - i])


def controlled_swap(circuit, control, target1, target2):
    circuit.cnot(target2, target1)
    circuit.ccx(control, target1, target2)
    circuit.cnot(target2, target1)


# ── N=15 hard-coded oracle ─────────────────────────────────────────────

def _controlled_mul_mod15(circuit, ctrl, work, a_power):
    w0, w1, w2, w3 = work
    if a_power % 15 == 1:
        pass
    elif a_power % 15 == 2:
        controlled_swap(circuit, ctrl, w3, w0)
        controlled_swap(circuit, ctrl, w3, w1)
        controlled_swap(circuit, ctrl, w3, w2)
    elif a_power % 15 == 4:
        controlled_swap(circuit, ctrl, w3, w1)
        controlled_swap(circuit, ctrl, w2, w0)
    elif a_power % 15 == 7:
        controlled_swap(circuit, ctrl, w0, w2)
        controlled_swap(circuit, ctrl, w1, w3)
        circuit.cnot(ctrl, w0); circuit.cnot(ctrl, w1)
        circuit.cnot(ctrl, w2); circuit.cnot(ctrl, w3)
    elif a_power % 15 == 8:
        controlled_swap(circuit, ctrl, w0, w3)
        controlled_swap(circuit, ctrl, w0, w2)
        controlled_swap(circuit, ctrl, w0, w1)
    elif a_power % 15 == 11:
        circuit.cnot(ctrl, w0); circuit.cnot(ctrl, w1)
        circuit.cnot(ctrl, w2); circuit.cnot(ctrl, w3)
        controlled_swap(circuit, ctrl, w0, w3)
        controlled_swap(circuit, ctrl, w0, w2)
        controlled_swap(circuit, ctrl, w0, w1)
    elif a_power % 15 == 13:
        circuit.cnot(ctrl, w0); circuit.cnot(ctrl, w1)
        circuit.cnot(ctrl, w2); circuit.cnot(ctrl, w3)
        controlled_swap(circuit, ctrl, w3, w0)
        controlled_swap(circuit, ctrl, w3, w1)
        controlled_swap(circuit, ctrl, w3, w2)
    elif a_power % 15 == 14:
        circuit.cnot(ctrl, w0); circuit.cnot(ctrl, w1)
        circuit.cnot(ctrl, w2); circuit.cnot(ctrl, w3)
    else:
        raise ValueError(f"Unsupported a_power mod 15 = {a_power % 15}")


def modular_exp_15(circuit, counting_qubits, work_qubits, a):
    n_count = len(counting_qubits)
    for j in range(n_count):
        a_power = pow(a, 2**j, 15)
        _controlled_mul_mod15(circuit, ctrl=counting_qubits[j],
                              work=work_qubits, a_power=a_power)


# ── General oracle ─────────────────────────────────────────────────────

def _multi_controlled_not(circuit, controls, target, ctrl_values, ancillas=None):
    for q, v in zip(controls, ctrl_values):
        if v == 0:
            circuit.x(q)
    n = len(controls)
    if n == 1:
        circuit.cnot(controls[0], target)
    elif n == 2:
        circuit.ccx(controls[0], controls[1], target)
    else:
        if ancillas is None or len(ancillas) < n - 2:
            raise ValueError(f"Need {n-2} ancilla qubits for {n}-controlled NOT")
        circuit.ccx(controls[0], controls[1], ancillas[0])
        for i in range(2, n):
            circuit.ccx(controls[i], ancillas[i - 2],
                        ancillas[i - 1] if i < n - 1 else target)
        if n > 3:
            for i in range(n - 2, 1, -1):
                circuit.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])
            circuit.ccx(controls[0], controls[1], ancillas[0])
        elif n == 3:
            circuit.ccx(controls[0], controls[1], ancillas[0])
    for q, v in zip(controls, ctrl_values):
        if v == 0:
            circuit.x(q)


def _controlled_transposition(circuit, ctrl, work, n_bits, val_i, val_j, ancilla_start):
    if val_i == val_j:
        return
    diff = val_i ^ val_j
    diff_bits = [b for b in range(n_bits) if (diff >> b) & 1]
    pivot = diff_bits[0]
    other_diff_bits = diff_bits[1:]
    pivot_val_in_i = (val_i >> pivot) & 1
    same_bits = [b for b in range(n_bits) if not ((diff >> b) & 1)]

    all_controls = [ctrl] + [work[b] for b in same_bits] + [work[pivot]]
    same_bit_values = [1] + [(val_i >> b) & 1 for b in same_bits]

    max_controls = len(all_controls)
    n_ancillas_needed = max(max_controls - 2, 0)
    ancillas = list(range(ancilla_start, ancilla_start + n_ancillas_needed)) if n_ancillas_needed > 0 else []

    for ob in other_diff_bits:
        ctrl_vals = same_bit_values + [pivot_val_in_i]
        _multi_controlled_not(circuit, all_controls, work[ob], ctrl_vals, ancillas)

    pivot_controls = [ctrl] + [work[b] for b in same_bits]
    pivot_ctrl_vals = [1] + [(val_i >> b) & 1 for b in same_bits]
    if len(other_diff_bits) > 0:
        pivot_controls_full = pivot_controls + [work[ob] for ob in other_diff_bits]
        pivot_ctrl_vals_full = pivot_ctrl_vals + [(val_j >> ob) & 1 for ob in other_diff_bits]
        n_anc2 = max(len(pivot_controls_full) - 2, 0)
        ancillas2 = list(range(ancilla_start, ancilla_start + n_anc2)) if n_anc2 > 0 else []
        _multi_controlled_not(circuit, pivot_controls_full, work[pivot], pivot_ctrl_vals_full, ancillas2)
    else:
        n_anc2 = max(len(pivot_controls) - 2, 0)
        ancillas2 = list(range(ancilla_start, ancilla_start + n_anc2)) if n_anc2 > 0 else []
        _multi_controlled_not(circuit, pivot_controls, work[pivot], pivot_ctrl_vals, ancillas2)

    for ob in other_diff_bits:
        ctrl_vals = same_bit_values + [1 - pivot_val_in_i]
        _multi_controlled_not(circuit, all_controls, work[ob], ctrl_vals, ancillas)


def _permutation_to_transpositions(perm):
    visited = set()
    transpositions = []
    for start in sorted(perm.keys()):
        if start in visited or perm[start] == start:
            visited.add(start)
            continue
        cycle = []
        x = start
        while x not in visited:
            visited.add(x)
            cycle.append(x)
            x = perm[x]
        for i in range(len(cycle) - 1, 0, -1):
            a_, b_ = cycle[0], cycle[i]
            transpositions.append((min(a_, b_), max(a_, b_)))
    return transpositions


def _controlled_mul_modN_general(circuit, ctrl, work, a_power, N, ancilla_start):
    ap = a_power % N
    if ap == 1:
        return
    n_bits = len(work)
    perm = {y: (y * ap) % N for y in range(N)}
    transpositions = _permutation_to_transpositions(perm)
    for (vi, vj) in transpositions:
        _controlled_transposition(circuit, ctrl, work, n_bits, vi, vj, ancilla_start)


def modular_exp_general(circuit, counting_qubits, work_qubits, a, N, ancilla_start):
    n_count = len(counting_qubits)
    for j in range(n_count):
        a_power = pow(a, 2**j, N)
        _controlled_mul_modN_general(circuit, ctrl=counting_qubits[j],
                                     work=work_qubits, a_power=a_power,
                                     N=N, ancilla_start=ancilla_start)


# ── Circuit builder ────────────────────────────────────────────────────

def shor_circuit(N, a, n_count=None):
    assert gcd(a, N) == 1, f"a={a} must be coprime to N={N}"
    n_work = ceil(log2(N))
    if n_count is None:
        n_count = 2 * n_work
    n_ancilla = max(n_work - 1, 0)
    total_qubits = n_count + n_work + n_ancilla
    counting_qubits = list(range(n_count))
    work_qubits = list(range(n_count, n_count + n_work))
    ancilla_start = n_count + n_work
    circuit = Circuit(total_qubits)
    for q in counting_qubits:
        circuit.h(q)
    circuit.x(work_qubits[0])
    if N == 15:
        modular_exp_15(circuit, counting_qubits, work_qubits, a)
    else:
        modular_exp_general(circuit, counting_qubits, work_qubits, a, N, ancilla_start)
    inverse_qft(circuit, counting_qubits)
    return circuit, counting_qubits, work_qubits


def pattern_to_networkx(pattern):
    nodes, edges = pattern.get_graph()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


# ── Timeout helper ─────────────────────────────────────────────────────

TIMEOUT_SECONDS = 10

class GraphTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise GraphTimeout("Timed out")


# ── Main: build and save ───────────────────────────────────────────────

def build_and_save(max_N=17, output_dir="shor_graph_database"):
    """Build all (N, a) graph states and save them to disk."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "full"), exist_ok=True)

    summary = []

    for N_val in range(3, max_N + 1):
        valid_bases = [a for a in range(2, N_val) if gcd(a, N_val) == 1]

        for a_val in valid_bases:
            r = 1
            while pow(a_val, r, N_val) != 1:
                r += 1

            tag = f"N{N_val}_a{a_val}"
            t0 = time.time()
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TIMEOUT_SECONDS)

            try:
                circ, cq, wq = shor_circuit(N_val, a_val)
                pat = circ.transpile().pattern
                G_full = pattern_to_networkx(pat)

                pat.standardize()
                pat.shift_signals()
                pat.perform_pauli_measurements()
                pat.standardize()
                G_reduced = pattern_to_networkx(pat)

                signal.alarm(0)
                elapsed = time.time() - t0

                # ── Save largest connected component of reduced graph ──
                largest_cc = max(nx.connected_components(G_reduced), key=len)
                G_largest = G_reduced.subgraph(largest_cc).copy()
                G_largest = nx.convert_node_labels_to_integers(G_largest)
                perm = list(range(G_largest.number_of_nodes()))
                random.shuffle(perm)
                G_largest = nx.relabel_nodes(G_largest, dict(enumerate(perm)))

                full_path = os.path.join(output_dir, "full", f"shor_{tag}.pkl")
                with open(full_path, "wb") as f:
                    pickle.dump(G_largest, f, protocol=pickle.HIGHEST_PROTOCOL)

                n_components = nx.number_connected_components(G_reduced)
                summary.append({
                    "N": N_val, "a": a_val, "order": r,
                    "nodes_full": G_full.number_of_nodes(),
                    "nodes_reduced": G_reduced.number_of_nodes(),
                    "nodes_saved": G_largest.number_of_nodes(),
                    "edges_saved": G_largest.number_of_edges(),
                    "num_components": n_components,
                    "time": elapsed,
                    "status": "ok",
                })

                print(f"{tag:>8}  r={r:<2}  "
                      f"Full: {G_full.number_of_nodes():>5}  "
                      f"Reduced: {G_reduced.number_of_nodes():>5} ({n_components} comps)  "
                      f"Saved largest: {G_largest.number_of_nodes():>5} nodes, "
                      f"{G_largest.number_of_edges()} edges  {elapsed:.1f}s")

            except GraphTimeout:
                signal.alarm(0)
                elapsed = time.time() - t0
                summary.append({
                    "N": N_val, "a": a_val, "order": r,
                    "status": "timeout", "time": elapsed,
                })
                print(f"{tag:>8}  r={r:<2}  SKIPPED (>{TIMEOUT_SECONDS}s)  {elapsed:.1f}s")

            except Exception as e:
                signal.alarm(0)
                elapsed = time.time() - t0
                summary.append({
                    "N": N_val, "a": a_val, "order": r,
                    "status": f"error: {e}", "time": elapsed,
                })
                print(f"{tag:>8}  r={r:<2}  ERROR: {e}  {elapsed:.1f}s")

            finally:
                signal.signal(signal.SIGALRM, old_handler)

    # ── Save summary ──
    summary_path = os.path.join(output_dir, "summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    ok = sum(1 for s in summary if s.get("status") == "ok")
    to = sum(1 for s in summary if s.get("status") == "timeout")
    er = len(summary) - ok - to
    print(f"\n{'='*60}")
    print(f"Total: {len(summary)} | OK: {ok} | Timeout: {to} | Error: {er}")
    print(f"Pickle files in:  {output_dir}/full/")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")

    return summary


def print_sample_graphs(output_dir="shor_graph_database", num_samples=5):
    """Load and print a few saved graphs to verify the database."""
    full_dir = os.path.join(output_dir, "full")

    print(f"\n{'='*60}")
    print("  Sample graphs from database")
    print(f"{'='*60}")

    pkl_files = sorted(f for f in os.listdir(full_dir) if f.endswith(".pkl"))
    sample = pkl_files[:num_samples]
    print(f"\n── Saved graphs ({len(pkl_files)} total, showing {len(sample)}) ──")
    for fname in sample:
        path = os.path.join(full_dir, fname)
        with open(path, "rb") as f:
            G = pickle.load(f)
        print(f"\n  {fname}")
        print(f"    Nodes ({G.number_of_nodes()}): {sorted(G.nodes())}")
        print(f"    Edges ({G.number_of_edges()}): {sorted(G.edges())[:15]}"
              f"{'...' if G.number_of_edges() > 15 else ''}")


if __name__ == "__main__":
    summ = build_and_save()
    print_sample_graphs()
