"""
Generate resource graph states for Deutsch-Jozsa algorithm (balanced oracle).

This script creates the MBQC resource graphs for Deutsch-Jozsa with n input qubits
ranging from 1 to 100, and saves them as pickle files for later use with NetworkX.
"""

from __future__ import annotations
import os
import pickle
import networkx as nx
from graphix import Circuit


def deutsch_jozsa_circuit(n, oracle_type='balanced', marked_qubits=None):
    """
    Build a Deutsch-Jozsa circuit for n input qubits.

    Parameters
    ----------
    n : int
        Number of input qubits (total qubits = n + 1 including ancilla).
    oracle_type : str
        'balanced'   - oracle flips ancilla for half the inputs.
        'constant_0' - oracle does nothing (f(x) = 0 for all x).
        'constant_1' - oracle flips ancilla unconditionally (f(x) = 1 for all x).
    marked_qubits : list[int] or None
        For 'balanced' oracle: which input qubits (0..n-1) are used in CNOTs.
        If None, defaults to all input qubits (a valid balanced function).

    Returns
    -------
    circuit : graphix.Circuit
    """
    total_qubits = n + 1
    ancilla = n

    circuit = Circuit(total_qubits)

    # Apply H to all input qubits
    for i in range(n):
        circuit.h(i)

    # Prepare ancilla in |-> state
    circuit.x(ancilla)
    circuit.h(ancilla)

    # Oracle
    if oracle_type == 'balanced':
        if marked_qubits is None:
            marked_qubits = list(range(n))
        for q in marked_qubits:
            circuit.cnot(q, ancilla)
    elif oracle_type == 'constant_0':
        pass
    elif oracle_type == 'constant_1':
        circuit.x(ancilla)
    else:
        raise ValueError(f"Unknown oracle_type '{oracle_type}'")

    # Final Hadamards on input qubits
    for i in range(n):
        circuit.h(i)

    return circuit


def pattern_to_networkx(pattern):
    """
    Extract the resource graph state from a Graphix pattern as a NetworkX graph.
    """
    nodes, edges = pattern.get_graph()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def generate_dj_graphs(n_min=1, n_max=1000, output_dir=None):
    """
    Generate Deutsch-Jozsa resource graphs for a range of input qubit counts.

    Parameters
    ----------
    n_min : int
        Minimum number of input qubits.
    n_max : int
        Maximum number of input qubits.
    output_dir : str or None
        Directory to save the graphs. If None, uses 'dj_graphs' in the script directory.

    Returns
    -------
    graphs : dict
        Dictionary mapping n -> NetworkX graph.
    """
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'dj_graphs')

    os.makedirs(output_dir, exist_ok=True)

    graphs = {}

    for n in range(n_min, n_max + 1):
        print(f"Generating Deutsch-Jozsa graph for n={n} input qubits...")

        circuit = deutsch_jozsa_circuit(n, oracle_type='balanced')
        pattern = circuit.transpile().pattern
        G = pattern_to_networkx(pattern)

        graphs[n] = G

        # Save individual graph as pickle
        graph_path = os.path.join(output_dir, f'dj_graph_n{n}.pkl')
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)

        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Connected: {nx.is_connected(G)}")

    # Save all graphs in one file for convenience
    all_graphs_path = os.path.join(output_dir, 'dj_graphs_all.pkl')
    with open(all_graphs_path, 'wb') as f:
        pickle.dump(graphs, f)

    print(f"\nAll graphs saved to: {output_dir}")
    print(f"  - Individual files: dj_graph_n1.pkl ... dj_graph_n{n_max}.pkl")
    print(f"  - Combined file: dj_graphs_all.pkl")

    return graphs


def load_dj_graph(n, input_dir=None):
    """
    Load a single Deutsch-Jozsa resource graph.

    Parameters
    ----------
    n : int
        Number of input qubits.
    input_dir : str or None
        Directory where graphs are stored. If None, uses 'dj_graphs' in the script directory.

    Returns
    -------
    G : networkx.Graph
    """
    if input_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'dj_graphs')

    graph_path = os.path.join(input_dir, f'dj_graph_n{n}.pkl')
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G


def load_all_dj_graphs(input_dir=None):
    """
    Load all Deutsch-Jozsa resource graphs.

    Parameters
    ----------
    input_dir : str or None
        Directory where graphs are stored. If None, uses 'dj_graphs' in the script directory.

    Returns
    -------
    graphs : dict
        Dictionary mapping n -> NetworkX graph.
    """
    if input_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'dj_graphs')

    all_graphs_path = os.path.join(input_dir, 'dj_graphs_all.pkl')
    with open(all_graphs_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs


if __name__ == '__main__':
    # Generate graphs for n=1 to n=100
    graphs = generate_dj_graphs(n_min=1, n_max=100)

    # Example: print summary
    print("\n--- Summary ---")
    for n, G in graphs.items():
        print(f"n={n}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
