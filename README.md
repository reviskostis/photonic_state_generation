# Photonic Graph State Generator

A Python toolkit for optimizing the generation of photonic graph states using quantum emitters. This repository provides algorithms for minimizing the number of emitters and CNOT gates required to generate arbitrary graph states, with applications in photonic quantum computing.

## Overview

Photonic graph states are essential resources for measurement-based quantum computing and quantum communication protocols. Generating these states efficiently or reducing the resource overhead is a key challenge. This toolkit addresses this by providing:

- **Emission ordering optimization** using simulated annealing (MinLA)
- **Edge reduction via local complementations** to simplify graph 
- **Dynamic programming-based merge ordering** for hierarchical graph decomposition
- **Circuit synthesis** from stabilizer tableaux using the algorithm from [Li et al. (2022)](https://doi.org/10.1038/s41534-022-00522-6)

## Features

- **Graph State Analysis**: Compute cut-rank, find modules/twin sets, and analyze graph properties over GF(2)
- **Edge Reduction**: Apply local complementation sequences to minimize edges while preserving the graph state LC-equivalence class
- **Emission Ordering**: Optimize photon emission order to minimize the number of required emitters
- **Circuit Generation**: Generate [Stim](https://github.com/quantumlib/Stim) circuits for graph state preparation
- **Clustering and Decomposition Algorithms**: Greedy, Kernighan-Lin based for clustering for hierarchical graph decomposition.

## Installation

### Prerequisites

- Python >= 3.9
- pip

### Dependencies

```bash
pip install numpy networkx matplotlib stim
```

### Optional Dependencies

For using some graphs relevant for quantum computing that do not exist in networkx:
```bash
pip install graphstate-opt
```

For ILP-based optimization with MOSEK solver:
```bash
pip install graphstate-opt[mosek]
```
Note: MOSEK requires a [license](https://docs.mosek.com/11.0/licensing/quickstart.html).

### Clone the Repository

```bash
git clone https://github.com/arr0w-hs/graph_state_optimization.git
cd graph_state_optimization
```

## Project Structure

```
photonic_state_generation/
├── photonic-graphstate-generator-main/    # Main codebase
│   ├── lib/                               # Core library modules
│   │   ├── circuitSolver.py               # Circuit generation algorithm
│   │   ├── generate_graph.py              # Graph utilities and MinLA
│   │   ├── tableau.py                     # Stabilizer tableau operations
│   │   ├── LC_edge_reduction.py           # Local complementation edge reduction
│   │   └── graphs.py                      # Graph type definitions
│   ├── utils/                             # Utility functions
│   │   ├── ufuncs.py                      # Standalone utility functions
│   │   └── zx_graph.py                    # ZX-calculus graph utilities
│   ├── data/                              # Entanglement class databases
│   ├── notebooks/                         # Analysis notebooks
│   ├── examples/                          # Example notebooks
│   ├── optimization_script.py             # Main optimization pipeline
│   ├── decompose_graph.py                 # Graph decomposition algorithms
│   ├── clustering.py                      # Clustering algorithms
│   └── workstation.py                     # Batch processing utilities
├── simulations_data/                      # Simulation results and logs
└── figures/                               # Generated figures and plots
```

## Quick Start

### Basic Usage

```python
import networkx as nx
from optimization_script import run_optimization

# Create a graph state (e.g., a 4-node linear cluster)
G = nx.path_graph(4)

# Run optimization with all features enabled
optimized_graph, circuit, stats = run_optimization(
    G,
    edge_reduction=True,   # Apply local complementation edge reduction
    minLA=True,            # Optimize emission ordering
    num_LA=10,             # Number of MinLA iterations
    opt_CNOT=True          # Optimize CNOT gate count
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

### Graph Decomposition with Clustering

```python
from clustering import GreedyClustering, KernighanLinClustering
import networkx as nx

G = nx.random_regular_graph(3, 10)

# Greedy clustering minimizes cut-rank
greedy = GreedyClustering()
cluster = greedy.do(G)

# Get the hierarchical inheritance tree
tree = cluster.get_inheritance()
print(f"Cluster hierarchy: {tree}")
```

## Key Modules

### `optimization_script.py`
Main optimization pipeline that combines edge reduction, emission ordering (MinLA), and CNOT optimization.

### `lib/circuitSolver.py`
Implements the time-reversed generation algorithm from [Li et al.](https://doi.org/10.1038/s41534-022-00522-6) with heuristic CNOT optimization.

### `lib/tableau.py`
Stabilizer tableau representation and manipulation for graph states, including echelon transform and height function computation.

### `lib/LC_edge_reduction.py`
Local complementation-based edge reduction using clustering coefficient heuristics. Includes the algorithm from [Ghanbari et al.](https://arxiv.org/abs/2401.00635).

### `decompose_graph.py`
Balanced rank-1 partitioning for graph decomposition. Computes cut-rank over GF(2) and finds modules/twin sets.

### `clustering.py`
Clustering algorithms for hierarchical graph decomposition:
- `GreedyClustering`: Merges clusters to minimize cut-rank
- `KernighanLinClustering`: Bisection-based clustering using Kernighan-Lin algorithm

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| `big_run_random_saminla.ipynb` | Benchmark comparisons on random Erdős-Rényi graphs |
| `dp_ordering.ipynb` | Hierarchical DP merge ordering with Reverse Cuthill-McKee |
| `examples/circuit scaling.ipynb` | Circuit scaling analysis |
| `examples/Full scale optimization.ipynb` | Complete optimization pipeline demo |
| `notebooks/eva_vs_ours_cluster.ipynb` | Comparison with cluster state methods |

## Data Files

The `data/` directory contains entanglement class databases:
- `combined_classes.txt`: Combined data from [Cabello et al.](https://arxiv.org/abs/0705.0998) for graphs ≤10 qubits
- `linear-rank-width_LC-classes.txt`: LC-class classification by number of emitters
- `entanglement_class_*.txt`: Individual entanglement classes by qubit count

## References

- Li et al. "Photonic resource state generation from a minimal number of quantum emitters." *npj Quantum Information* 8, 11 (2022). [DOI](https://doi.org/10.1038/s41534-022-00522-6)
- Ghanbari et al. "Optimization of graph state generation." *arXiv:2401.00635* (2024). [arXiv](https://arxiv.org/abs/2401.00635)
- Oum & Seymour. "Rank-width and vertex-minors." *Journal of Combinatorial Theory, Series B* 95, 79-100 (2005).
- Cabello et al. "Optimal preparation of graph states." *Physical Review A* 83, 012314 (2011).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

To be decided

## Citation

To be decided


SPEEDUP ideas:
- store the clustering coefficient somewhere, only modify them if we apply a local complementation on itself or one of its neighbor, we thus don't need to compute them all the time.
- For simulation, make a quick check whether we reach a theoretical maximum in the optimizations:
    - For example, if n_edges = n_nodes - 1 (the theoretical minimum for connected graphs)
    - or crossing edges = cutrank
    - It could be particularly useful for the simulated annealing in the case of small graphs
    - tabu search if we already went to the same graph twice (can be a break point in fast greedy algorithm)
    - Look up in the Adcock et al. list


Project:
- find the minimum number of CZ to make any graph with less than 12 nodes