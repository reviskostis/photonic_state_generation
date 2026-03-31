# Photonic Graph State Generatorion: Optimizing the number of emitters

A Python codebase for optimizing the generation of photonic graph states using quantum emitters. This repository provides algorithms for minimizing the number of emitters and CNOT gates required to generate arbitrary graph states, with applications in photonic quantum computing and measurement-based quantum computation (MBQC).

> **Acknowledgement:** This repository was based on a codebase shared privately by **Nils Tomke Ottink**, developed during his master thesis. The original code provided the foundation for the stabilizer-tableau manipulation, circuit generation, local-complementation edge-reduction, as well as the SAminLA algorithm for optimzing the number of emitterrs. Subsequent development including the rest of the algorithms for optimizing the number of emitters including rank_width, rank_width_sa and path_clustering as well as the preparation of plenty of datatsets for testing and comparing the afforementioned algorithms was carried afterwards. 


## Overview

Photonic graph states are essential resources for measurement-based quantum computing and quantum communication protocols. Generating these states efficiently, so minimising the number of quantum emitters and entangling gates is a key practical challenge. 

- **Emission ordering optimization** Four main algorithms are implemented. Namely: SAminLA, rank_width, rank_width_sa and path_clustering
- **Edge reduction via local complementations** to simplify graphs while preserving LC-equivalence
- **Circuit synthesis** from stabilizer tableaux using the time-reversed generation algorithm of [Li et al. (2022)](https://www.nature.com/articles/s41534-022-00522-6)
- **Benchmark pipelines**: Compare the number of emitters with the codebase developed by [Takou et al. (2022)](https://www.nature.com/articles/s41534-025-01056-3)

## Features

- **Graph State Analysis**: Compute cut-rank over GF(2), find modules/twin sets, and analyse graph properties relevant to resource-state generation
- **Edge Reduction**: Apply local complementation sequences to minimise edges while preserving the graph state's LC-equivalence class
- **Emission Ordering**: Multiple ordering strategies — spectral (Fiedler), Reverse Cuthill-McKee, minimum-degree, and simulated annealing — to minimise the required number of emitters
- **Linear Rank-Width Approximation**: Greedy heuristics and SA-based refinement for computing near-optimal linear rank-width orderings
- **Path-Based Clustering**: Polynomial-time heuristic that decomposes graphs via longest-path extraction, inter-cluster Traveling Salesman Problem (TSP), and boundary SA refinement
- **Circuit Generation**: Generate [Stim](https://github.com/quantumlib/Stim) circuits for graph state preparation, with optional ZX-calculus optimisation via [PyZX](https://github.com/Quantomatic/pyzx)
- **Clustering and Decomposition**: Greedy and Kernighan-Lin clustering for hierarchical graph decomposition with cut-rank minimisation
- **MBQC Graph Generation**: Generate resource-state graphs for Shor's algorithm and Deutsch-Jozsa via [Graphix](https://github.com/TeamGraphix/graphix)
- **Benchmark Infrastructure**: Parallelised benchmark runners for QECC, RHG, Shor, and random graph databases
- **Algorithm comparison**: Detailed comparison of every pipeline step to address how modifying each one affects the individual algorithm performance. Compare the algorithms with each other for their best version. The metrics used were the number of emitters, the number of emitter CNOTs, the total gatecount, and the runtime in our machine.

## Installation

### Prerequisites

- Python >= 3.9
- pip

### Dependencies

```bash
pip install numpy networkx matplotlib stim
```

### Optional Dependencies

For ZX-calculus circuit optimisation:
```bash
pip install pyzx cirq
```

For MBQC pattern generation (Shor, Deutsch-Jozsa graphs):
```bash
pip install graphix
```

### Clone the Repository

```bash
git clone https://github.com/arr0w-hs/graph_state_optimization.git
cd graph_state_optimization
```

## Project Structure

```
photonic_state_generation/
├── photonic-graphstate-generator-main/   # Main codebase
│   ├── lib/                              # Core library modules
│   │   ├── circuitSolver.py              # Time-reversed circuit generation (Li et al. 2022)
│   │   ├── generate_graph.py             # Graph-state utilities and MinLA SA
│   │   ├── tableau.py                    # Stabilizer tableau representation & operations
│   │   ├── LC_edge_reduction.py          # Local complementation edge reduction
│   │   ├── graphs.py                     # Predefined graph types & encoded graph-codes
│   │   └── typeIIfusions.py              # Type II fusion gate primitives
│   ├── utils/                            # Utility functions
│   │   ├── ufuncs.py                     # Standalone tableau and height-function helpers
│   │   └── zx_graph.py                   # ZX-calculus optimisation (Stim and PyZX)
│   ├── data/                             # Entanglement class databases (less than 10 qubits)
│   ├── tests/                            # Unit tests for core modules
│   ├── examples/                         # Example and demo notebooks
│   ├── notebooks/                        # Analysis and comparison notebooks
│   ├── optimization_script.py            # Main optimisation pipeline for circuit extraction and SaminLA algorithm
│   ├── path_clustering.py               # Path-based LRW approximation
│   ├── graph_utils.py                    # GF(2) linear algebra & cut-rank
│   ├── rank_width.py                     # LRW via greedy heuristics (spectral, RCM, min-degree)
│   ├── rank_width_sa.py                  # LRW via simulated annealing
│   ├── qecc_runs.py                      # QECC benchmark runner
│   ├── rhg_runs.py                       # RHG lattice benchmark runner
│   ├── runs_shor_full.py                 # Shor algorithm benchmark runner
│   └── save_shor_graphs.py              # Shor resource-state graph generator
├── simulations_data/                     # Benchmark results and logs
│   ├── step2_results/                    # Phase 2 (edge reduction) results
│   ├── step4_results/                    # Phase 4 (full pipeline) results
│   ├── step4_results_rhg/                # RHG lattice results
│   ├── step4_results_shor_red/           # Shor algorithm results
│   └── *.txt / *.pkl                     # Per-algorithm metric traces
└── figures/                              # Generated figures and plots
    ├── algos_comparison/                 # Algorithm comparison plots
    ├── cluster_states/                   # Cluster state visualisations
    ├── effect_of_edge_reduction/         # Edge reduction analysis
    └── step3/ , step4/                   # Pipeline stage plots
```

## Quick Start

### Basic Usage

```python
import networkx as nx
from optimization_script import run_optimization

# Create a graph state (e.g., a 4-node linear cluster)
G = nx.path_graph(4)

# Run optimisation with all features enabled
optimized_graph, circuit, stats = run_optimization(
    G,
    edge_reduction=True,   # Apply local complementation edge reduction
    minLA=True,            # Optimise emission ordering
    num_LA=10,             # Number of MinLA iterations
    opt_CNOT=True          # Optimise CNOT gate count
)

# stats contains: [num_emitters, num_emitter_cnots, num_gates, emission_order, LC_sequence]
print(f"Number of emitters required: {stats[0]}")
print(f"Number of emitter CNOTs: {stats[1]}")
```

### Edge Reduction

```python
from lib.LC_edge_reduction import LCReduction
from lib.generate_graph import GraphstateGenerator
import networkx as nx

# Create a graph
G = nx.grid_2d_graph(3, 3)
G = nx.convert_node_labels_to_integers(G)

# Apply edge reduction
lcr = LCReduction(G)
lcr.edge_reduction(with_increase=True)
lc_sequence = lcr.get_path()

# Apply the LC sequence
reduced_graph = G.copy()
for node in lc_sequence:
    reduced_graph = GraphstateGenerator.local_complementation(reduced_graph, node)

print(f"Original edges: {G.number_of_edges()}")
print(f"Reduced edges: {reduced_graph.number_of_edges()}")
```

### Linear Rank-Width Ordering

```python
from rank_width_sa import simulated_annealing
from rank_width import evaluate_ordering, spectral_ordering
import networkx as nx

G = nx.random_regular_graph(3, 20)

# Phase A: initial ordering via spectral method
ordering = spectral_ordering(G)
lrw = evaluate_ordering(G, ordering)
print(f"Spectral LRW: {lrw}")

# Phase C: refine with simulated annealing
best_ordering, best_cost = simulated_annealing(G, ordering)
print(f"SA-refined LRW: {best_cost}")
```

### Path-Based Clustering

```python
from path_clustering import path_clustering_lrw
import networkx as nx

G = nx.random_regular_graph(3, 30)

# Polynomial-time LRW approximation via path clustering
ordering, cost = path_clustering_lrw(G)
print(f"Path clustering LRW: {cost}")
```


## Key Modules

### Core Library (`lib/`)

| Module | Description |
|--------|-------------|
| `circuitSolver.py` | Time-reversed graph-state generation algorithm from [Li et al. (2022)](https://doi.org/10.1038/s41534-022-00522-6). Iterates photons in reverse order, applies height-function logic, and optimises emitter selection to minimise Pauli corrections. Outputs Stim circuits. |
| `tableau.py` | Stabilizer tableau (n×2n boolean matrix over GF(2)) with row reduction, height-function computation, Clifford gate application via Stim, and sign tracking. |
| `generate_graph.py` | `GraphstateGenerator` class for adjacency-to-tableau conversion, node reordering, local complementation, MinLA simulated annealing, and leaf contraction. Includes predefined graphs (ring, Shor 9-qubit, cube, snowflake). |
| `LC_edge_reduction.py` | `LCReduction` class for edge minimisation via local complementations. Uses clustering-coefficient heuristics. |
| `graphs.py` | Predefined benchmark graphs (Shor, repeater, cluster families) and `encodedGraphs` class for logical graph-code encoding. |
| `typeIIfusions.py` | Type II fusion gate primitives for photonic MBQC. |

### Ordering & Decomposition Algorithms

| Module | Description |
|--------|-------------|
| `optimization_script.py` | Main pipeline orchestrating edge reduction, SAminLA ordering, and CNOT optimisation. |
| `rank_width.py` | Linear rank-width approximation via spectral (Fiedler), Reverse Cuthill-McKee, and minimum-degree orderings. |
| `rank_width_sa.py` | Simulated annealing for LRW with swap/reverse/relocate moves and geometric cooling. |
| `path_clustering.py` | Polynomial-time LRW heuristic: longest-path extraction -> cluster construction -> inter-cluster TSP -> intra-cluster ordering -> boundary SA refinement. |
| `graph_utils.py` | GF(2) linear algebra: biadjacency matrices, cut-rank, and Gaussian elimination over GF(2). |

### Utilities

| Module | Description |
|--------|-------------|
| `utils/ufuncs.py` | Standalone helpers for tableau ↔ Pauli-string conversion, echelon transform, height function, and bigram extraction. |
| `utils/zx_graph.py` | ZX-calculus integration: Stim ↔ Cirq ↔ PyZX conversion and basic ZX optimisations. |

### Benchmark Runners

| Module | Description |
|--------|-------------|
| `qecc_runs.py` | Parallelised benchmarks on QECC graph databases with all four algorithms (LRW, LRW+SA, path clustering, SA+MinLA). |
| `rhg_runs.py` | Benchmarks on RHG (Raussendorf-Harrington-Goyal) lattice graphs. |
| `runs_shor_full.py` | Benchmarks on Shor algorithm resource-state graphs. |
| `save_shor_graphs.py` | Generates Shor algorithm MBQC graphs for all valid (N, a) pairs using Graphix, including QFT, modular exponentiation, and controlled-swap circuits. |
| `benchmark_algos.ipynb` | Calculate the number of emitters for circles and caterpillar graphs.|

## Tests

Unit tests are located in `photonic-graphstate-generator-main/tests/`:

```bash
cd photonic-graphstate-generator-main
python -m pytest tests/
```

- `test_circuitSolver.py` — Circuit generation correctness
- `test_generate_graph.py` — Graph manipulation and MinLA
- `test_tableau.py` — Stabilizer tableau operations

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| `examples/Full scale optimization.ipynb` | Complete optimisation pipeline demo |
| `examples/circuit scaling.ipynb` | Circuit scaling analysis |
| `examples/Emission_circuit_RUS.ipynb` | Emission circuit with repeat-until-success |
| `examples/generate_encoded_graph.ipynb` | Encoded graph-code generation |
| `examples/ZX_library.ipynb` | ZX-calculus optimisation demo |
| `graphix_shor.ipynb` | Shor algorithm MBQC pattern generation |
| `graphix_deutsch_jozsa.ipynb` | Deutsch-Jozsa MBQC pattern generation |
| `edge_reduction_comparison.ipynb` | LC edge reduction effectiveness analysis |
| `eva_vs_ours_cluster.ipynb` | Comparison of SaminLA with Takou's et al. implementation. |
| `ghanbari_vs_ours_table.ipynb` | Comparison of SAminLA with [Ghanbari and Hoi-Kwong Lo (2025)](https://arxiv.org/pdf/2509.22777) |
| `qec_and_rhg_plotting.ipynb` | QECC and RHG benchmark result plots |
| `large_algo_comparison_plotting.ipynb` | Large-scale comparison of the best version of the final algorithms with the random labelling as well as best algorithm determination. |

## Data

### Entanglement Class Databases (`data/`)
- `combined_classes.txt` — Combined data from [Cabello et al.](https://arxiv.org/abs/0705.0998) for graphs ≤10 qubits
- `linear-rank-width_LC-classes.txt` — LC-class classification by number of emitters
- `entanglement_class_*.txt` — Individual entanglement classes by qubit count (2–10)

### Graph Databases
- `graphs_database_paul/` — QECC graph collection
- `shor_graph_database/` — Shor algorithm resource-state graphs
- `step4_rgh_database/` — RHG lattice graphs
- `cater_database/` - Catepilar graphs database
-`circle_database/` - Circle graphs database
- `dj_graphs/` — Deutsch-Jozsa algorithm graphs
- `random_graph_db/` — Random Erdos-Renyi graph instances

### Simulation Results (`simulations_data/`)
Benchmark outputs organised by algorithm and graph family, including emitter counts, CNOT counts, and node sizes for all algorithm variants (path clustering, rank width, rank width + SA, SA + MinLA).

## References

- Li et al. "Photonic resource state generation from a minimal number of quantum emitters." [Li et al. (2022)](https://www.nature.com/articles/s41534-022-00522-6)
- Takou et al. "Optimization complexity and resource minimization of emitter-based photonic graph state generation protocols" [Takou et al. (2025)](https://www.nature.com/articles/s41534-025-01056-3)

## Acknowledgements

This project builds upon a codebase originally developed and shared privately by **Nils Tomke Ottink**. The original code was developed for his Master thesis . The original code provided the core stabilizer-tableau infrastructure, circuit generation algorithm, and local-complementation edge-reduction methods.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

Will be added after the paper is uploaded in arxiv

