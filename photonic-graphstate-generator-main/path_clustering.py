"""
Path-Based Clustering for Linear Rank-Width Minimisation
==========================================================

Approximates the Linear Rank-Width (lrw) of a graph G = (V, E) by
hierarchically decomposing it into clusters along longest paths and then
finding good orderings within and across clusters.

Algorithm Outline
-----------------
1. Path extraction: Find a long (simple) path P_1 in G using fast
   polynomial-time heuristics (BFS-diameter + greedy DFS, not exact
   backtracking which is NP-hard).  Every vertex v on P_1, together
   with its neighbourhood N(v), defines the first cluster G_{P_1}.

2. Recursive clustering: Remove G_{P_1} from G.  For each connected
   component of G \\ G_{P_1}, repeat step 1 to obtain clusters
   G_{P_2^1}, ..., G_{P_2^{n_2}}, etc.  Continue until no vertices
   remain (or only isolated vertices).

3. Cluster-graph construction: Build a weighted meta-graph where
   each node is a cluster and edge weights w_{ij} count the number of
   shared neighbourhood elements between pairs of clusters.

4. Cluster ordering (inter-cluster):
   * If <= 6 clusters: solve the TSP exactly (brute-force permutations).
   * Otherwise: simulated-annealing TSP.

5. Intra-cluster ordering:
   * If a cluster has <= 6 vertices: exact brute-force over all
     permutations, using the height function as cost.
   * Otherwise: call the SA-based solver from ``rank_width_sa.py``.

6. Assembly: Concatenate the intra-cluster orderings in the
   determined inter-cluster order to form a global ordering.

7. Global SA refinement: Run a final SA pass on the full ordering
   where moves preferentially target "boundary" vertices (those
   contributing to inter-cluster weights) to fine-tune the result.

Cost Function
-------------
Throughout, the cost of an ordering is ``max(h)`` the maximum of the
height function of the stabilizer tableau which equals the number of
emitters for graph-state generation and, at optimality, equals the
linear rank-width.

Usage
-----
    from path_clustering import path_clustering_lrw

    lrw, ordering = path_clustering_lrw(G, verbose=True)

Dependencies
------------
* ``rank_width_sa.py`` SA infrastructure (``_full_evaluate``,
  ``simulated_annealing``, ``gf2_rank``, initial-ordering generators,
  relabelling helpers, and Metropolis-based SA loop).
"""

from __future__ import annotations

import math
import random
from itertools import permutations
from typing import Optional

import networkx as nx
import numpy as np

# --- Re-use infrastructure from rank_width_sa ---------------------------
from rank_width_sa import (
    _full_evaluate,
    _relabel_to_int,
    evaluate_ordering,
    gf2_rank,
    simulated_annealing,
    generate_initial_orderings,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 — Long-path finder (polynomial-time heuristic)
# ═══════════════════════════════════════════════════════════════════════════
#
# Finding the longest simple path is NP-hard in general graphs.
#
# We use polynomial-time heuristics:
#   1. BFS-diameter endpoints (two rounds of BFS).
#   2. Greedy DFS extension: from a starting vertex, always extend to
#      the *unvisited* neighbour with the highest remaining degree
#      (breaking ties randomly).  This is O(V + E) per attempt.
#   3. Multi-start: repeat from several starting vertices (BFS-far
#      endpoints, random samples, high-degree vertices) and keep the
#      longest path found.
#
# This trades optimality of the path for speed: the paths are typically
# close to the longest and are always long enough for effective
# clustering.  Total cost: O(k · (V + E)) with k = O(1) restarts.
# ═══════════════════════════════════════════════════════════════════════════


def _bfs_farthest(graph: nx.Graph, source: int) -> int:
    """Return the vertex farthest from *source* by BFS (unweighted).

    Runs in O(V + E).
    """
    visited = {source}
    queue = [source]
    last = source
    while queue:
        nxt = []
        for v in queue:
            for u in graph.neighbors(v):
                if u not in visited:
                    visited.add(u)
                    nxt.append(u)
                    last = u
        queue = nxt
    return last


def _greedy_dfs_path(graph: nx.Graph, start: int, rng: random.Random | None = None) -> list[int]:
    """Greedy DFS from start: always extend to the unvisited neighbour
    with the highest degree in the residual graph.  Ties are broken
    randomly when rng is provided, otherwise by node index.

    Runs in O(V + E) — no backtracking.
    """
    path = [start]
    visited = {start}
    current = start

    while True:
        # Collect unvisited neighbours
        nbrs = [u for u in graph.neighbors(current) if u not in visited]
        if not nbrs:
            break
        # Pick the neighbour with the most unvisited neighbours itself
        # (greedy: prefer vertices that keep options open).
        def _score(u):
            return sum(1 for w in graph.neighbors(u) if w not in visited)

        if rng is not None:
            # Shuffle first so that ties are broken randomly
            rng.shuffle(nbrs)
        nbrs.sort(key=_score, reverse=True)
        nxt = nbrs[0]
        visited.add(nxt)
        path.append(nxt)
        current = nxt

    return path


def longest_path(graph: nx.Graph, n_random_starts: int = 6) -> list[int]:
    """Return a long simple path in graph using polynomial heuristics.

    Strategy (all O(V + E) per attempt):
    1. BFS-diameter: two rounds of BFS give approximate diameter
       endpoints; greedy-DFS from both.
    2. High-degree seeds: greedy-DFS from the top-3 highest-degree
       vertices.
    3. Random seeds: greedy-DFS from n_random_starts randomly chosen
       vertices.

    The longest path found across all attempts is returned.

    Total complexity: O((n_random_starts + 5) · (V + E))  —  polynomial.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return []
    if n == 1:
        return list(graph.nodes())

    nodes = list(graph.nodes())
    rng = random.Random(42)  # deterministic tie-breaking

    best: list[int] = []

    # --- Candidate start vertices ---
    starts: list[int] = []

    # BFS-diameter endpoints
    s0 = nodes[0]
    far1 = _bfs_farthest(graph, s0)
    far2 = _bfs_farthest(graph, far1)
    starts.extend([far1, far2])

    # High-degree vertices (top 3)
    deg_sorted = sorted(nodes, key=lambda v: graph.degree(v), reverse=True)
    starts.extend(deg_sorted[:3])

    # Random vertices
    if n > 5:
        starts.extend(rng.sample(nodes, min(n_random_starts, n)))

    # De-duplicate while preserving order
    seen: set[int] = set()
    unique_starts: list[int] = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            unique_starts.append(s)

    # --- Run greedy DFS from each start ---
    for s in unique_starts:
        p = _greedy_dfs_path(graph, s, rng)
        if len(p) > len(best):
            best = p
        # Also try extending from the other end of p
        p_rev = _greedy_dfs_path(graph, p[-1], rng)
        if len(p_rev) > len(best):
            best = p_rev

    return best


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — Cluster extraction 
# ═══════════════════════════════════════════════════════════════════════════

class PathCluster:
    """One cluster produced by the path-peeling decomposition.

    Attributes
    ----------
    core_path : list[int]
        Vertices on the longest path that seeds this cluster.
    vertices : set[int]
        All vertices in the cluster: core path u their neighbours.
    level : int
        Recursion depth (0 = first / primary path).
    """

    def __init__(self, core_path: list[int], vertices: set[int], level: int):
        self.core_path = core_path
        self.vertices = vertices
        self.level = level

    def __repr__(self):
        return (
            f"PathCluster(level={self.level}, "
            f"|core|={len(self.core_path)}, |V|={len(self.vertices)})"
        )


def _build_cluster_from_path(
    graph: nx.Graph,
    path: list[int],
    already_claimed: set[int],
) -> PathCluster:
    """Construct a cluster from a core path.

    For each vertex on the path, include all its neighbours that have
    *not* already been claimed by a higher-level cluster.  Neighbours
    that belong to a previously-found core path are ignored (as per the
    user's decision to skip them).
    """
    cluster_vertices = set(path)
    for v in path:
        for u in graph.neighbors(v):
            if u not in already_claimed:
                cluster_vertices.add(u)
    return PathCluster(
        core_path=list(path),
        vertices=cluster_vertices,
        level=0,  # will be set by the caller
    )


def extract_clusters(graph: nx.Graph, verbose: bool = False) -> list[PathCluster]:
    """Hierarchical path-peeling decomposition of graph.

    Returns a list of ``PathCluster`` objects covering every vertex of
    graph.  The first cluster corresponds to the primary longest path
    and its neighbourhood, and subsequent clusters are obtained
    recursively from the residual graph.
    """
    clusters: list[PathCluster] = []
    all_vertices = set(graph.nodes())
    claimed: set[int] = set()          # vertices assigned to some cluster
    level = 0

    remaining_vertices = set(all_vertices)

    while remaining_vertices:
        # Induce subgraph on unclaimed vertices
        H = graph.subgraph(remaining_vertices).copy()

        # Handle isolated vertices: group them into one trivial cluster
        # (they don't affect the ordering cost).
        isolates = set(nx.isolates(H))
        non_isolate_vertices = remaining_vertices - isolates

        if not non_isolate_vertices:
            # Only isolated vertices left
            if isolates:
                for iso in isolates:
                    clusters.append(PathCluster(
                        core_path=[iso],
                        vertices={iso},
                        level=level,
                    ))
                    claimed.add(iso)
                remaining_vertices -= isolates
            break

        # Work on each connected component separately
        components_processed = False
        for comp_nodes in list(nx.connected_components(H.subgraph(non_isolate_vertices))):
            if len(comp_nodes) == 1:
                v = next(iter(comp_nodes))
                clusters.append(PathCluster(
                    core_path=[v],
                    vertices={v},
                    level=level,
                ))
                claimed.add(v)
                remaining_vertices.discard(v)
                components_processed = True
                continue

            comp = graph.subgraph(comp_nodes).copy()
            path = longest_path(comp)

            if verbose:
                print(
                    f"  Level {level}: longest path length {len(path)} "
                    f"in component of size {len(comp_nodes)}"
                )

            cluster = _build_cluster_from_path(graph, path, claimed)
            cluster.level = level
            clusters.append(cluster)

            claimed |= cluster.vertices
            remaining_vertices -= cluster.vertices
            components_processed = True

        # Also claim any isolated vertices at this level
        if isolates:
            for iso in isolates:
                clusters.append(PathCluster(
                    core_path=[iso],
                    vertices={iso},
                    level=level,
                ))
                claimed.add(iso)
            remaining_vertices -= isolates

        if not components_processed:
            break  # safety valve

        level += 1

    if verbose:
        print(f"  Total clusters: {len(clusters)}, covering {len(claimed)}/{len(all_vertices)} vertices")

    return clusters


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 — Cluster meta-graph (inter-cluster weights)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_cluster_weights(
    graph: nx.Graph,
    clusters: list[PathCluster],
) -> dict[tuple[int, int], int]:
    """Compute inter-cluster edge weights.

    For every pair (i, j) of clusters, w_{ij} counts the number of edges
    in the original graph that connect a vertex in cluster i to a vertex
    in cluster j.
    """
    # Map vertex -> cluster index
    vertex_to_cluster: dict[int, int] = {}
    for idx, cl in enumerate(clusters):
        for v in cl.vertices:
            vertex_to_cluster[v] = idx

    weights: dict[tuple[int, int], int] = {}
    for u, v in graph.edges():
        ci = vertex_to_cluster.get(u)
        cv = vertex_to_cluster.get(v)
        if ci is None or cv is None:
            continue
        if ci == cv:
            continue
        key = (min(ci, cv), max(ci, cv))
        weights[key] = weights.get(key, 0) + 1

    return weights


def _build_cluster_graph(
    clusters: list[PathCluster],
    weights: dict[tuple[int, int], int],
) -> nx.Graph:
    """Build a networkx weighted graph over cluster indices."""
    CG = nx.Graph()
    CG.add_nodes_from(range(len(clusters)))
    for (i, j), w in weights.items():
        CG.add_edge(i, j, weight=w)
    return CG


def _boundary_vertices(
    graph: nx.Graph,
    clusters: list[PathCluster],
) -> set[int]:
    """Return the set of vertices that have at least one neighbour in a
    different cluster.  These are the "interface" vertices whose
    positions most affect the global ordering cost.
    """
    vertex_to_cluster: dict[int, int] = {}
    for idx, cl in enumerate(clusters):
        for v in cl.vertices:
            vertex_to_cluster[v] = idx

    boundary: set[int] = set()
    for u, v in graph.edges():
        ci = vertex_to_cluster.get(u)
        cv = vertex_to_cluster.get(v)
        if ci is not None and cv is not None and ci != cv:
            boundary.add(u)
            boundary.add(v)
    return boundary


# ═══════════════════════════════════════════════════════════════════════════
#  Section 4 — Inter-cluster ordering (TSP on cluster graph)
# ═══════════════════════════════════════════════════════════════════════════

def _tsp_exact(dist_matrix: np.ndarray) -> list[int]:
    """Solve a small TSP exactly (brute-force over all permutations).

    dist_matrix is a symmetric n times n matrix of pairwise distances.
    We fix node 0 as the start (the primary cluster) and permute the rest.

    Returns the best permutation of range(n) starting with 0.
    """
    n = dist_matrix.shape[0]
    if n <= 1:
        return list(range(n))
    if n == 2:
        return [0, 1]

    best_cost = float("inf")
    best_perm: list[int] = list(range(n))

    # Fix cluster 0 first, permute the rest
    others = list(range(1, n))
    for perm in permutations(others):
        order = [0] + list(perm)
        cost = sum(dist_matrix[order[i], order[i + 1]] for i in range(n - 1))
        if cost < best_cost:
            best_cost = cost
            best_perm = list(order)
    return best_perm


def _tsp_sa(
    dist_matrix: np.ndarray,
    *,
    T_start: float = 5.0,
    T_min: float = 0.01,
    alpha: float = 0.95,
    steps_per_temp: int = 100,
    seed: Optional[int] = None,
) -> list[int]:
    """Solve the TSP via simulated annealing on a distance matrix.

    Fixes node 0 as the first cluster and permutes the remaining nodes.
    """
    rng = random.Random(seed)
    n = dist_matrix.shape[0]
    if n <= 1:
        return list(range(n))

    # Initial order: nearest-neighbour heuristic starting from 0
    visited = {0}
    order = [0]
    for _ in range(n - 1):
        last = order[-1]
        best_next = None
        best_d = float("inf")
        for j in range(n):
            if j not in visited and dist_matrix[last, j] < best_d:
                best_d = dist_matrix[last, j]
                best_next = j
        if best_next is None:
            # Disconnected – pick any unvisited
            best_next = (set(range(n)) - visited).pop()
        visited.add(best_next)
        order.append(best_next)

    def _path_cost(o: list[int]) -> float:
        return sum(dist_matrix[o[i], o[i + 1]] for i in range(len(o) - 1))

    cur_cost = _path_cost(order)
    best_cost = cur_cost
    best_order = order[:]

    T = T_start
    while T > T_min:
        for _ in range(steps_per_temp):
            # Swap two positions (not position 0 – cluster 0 stays first)
            i = rng.randint(1, n - 1)
            j = rng.randint(1, n - 1)
            while j == i:
                j = rng.randint(1, n - 1)
            order[i], order[j] = order[j], order[i]
            new_cost = _path_cost(order)
            delta = new_cost - cur_cost
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_order = order[:]
            else:
                order[i], order[j] = order[j], order[i]  # undo
        T *= alpha

    return best_order


def order_clusters(
    clusters: list[PathCluster],
    cluster_graph: nx.Graph,
    *,
    exact_threshold: int = 6,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> list[int]:
    """Determine the ordering of clusters (a permutation of cluster indices).

    * If <= exact_threshold clusters: exact TSP (brute-force).
    * Otherwise: SA-based TSP.

    The distance between two clusters is the *negative* of their weight
    (we want heavily-connected clusters to be adjacent).
    """
    k = len(clusters)
    if k <= 1:
        return list(range(k))

    # Build distance matrix: d(i,j) = -weight(i,j) so that TSP minimisation
    # keeps heavy pairs close.  For non-adjacent clusters use a large value.
    max_w = 1
    for _, _, data in cluster_graph.edges(data=True):
        w = data.get("weight", 1)
        if w > max_w:
            max_w = w

    dist = np.full((k, k), float(max_w + 1), dtype=float)
    np.fill_diagonal(dist, 0.0)
    for i, j, data in cluster_graph.edges(data=True):
        w = data.get("weight", 1)
        # Invert: high weight → small distance
        d = max_w + 1 - w
        dist[i, j] = d
        dist[j, i] = d

    if k <= exact_threshold:
        order = _tsp_exact(dist)
    else:
        order = _tsp_sa(dist, seed=seed)

    if verbose:
        print(f"  Cluster ordering: {order}")
    return order


# ═══════════════════════════════════════════════════════════════════════════
#  Section 5 — Intra-cluster ordering
# ═══════════════════════════════════════════════════════════════════════════

def _exact_ordering(graph: nx.Graph, vertices: list[int]) -> list[int]:
    """Find the ordering of vertices that minimises the height-function
    cost (number of emitters) by exhaustive enumeration.

    Only feasible for |vertices| <= 6 (6! = 720 permutations).
    """
    if len(vertices) <= 1:
        return list(vertices)

    subgraph = graph.subgraph(vertices).copy()
    # Relabel to 0..n-1 for evaluation
    mapping = {v: i for i, v in enumerate(vertices)}
    inv_mapping = {i: v for i, v in enumerate(vertices)}
    G_int = nx.relabel_nodes(subgraph, mapping)
    n = G_int.number_of_nodes()

    adj = nx.adjacency_matrix(G_int).toarray().astype(np.uint8)

    best_cost = n + 1
    best_order: list[int] = list(range(n))

    for perm in permutations(range(n)):
        perm_list = list(perm)
        cost, _ = _full_evaluate(adj, perm_list)
        if cost < best_cost:
            best_cost = cost
            best_order = perm_list[:]
            if best_cost == 0:
                break

    # Map back to original labels
    return [inv_mapping[i] for i in best_order]


def _sa_ordering(
    graph: nx.Graph,
    vertices: list[int],
    *,
    seed: Optional[int] = None,
    verbose: bool = False,
    T_start: float = 4.0,
    T_min: float = 0.01,
    alpha: float = 0.97,
    steps_per_temp: int = 30,
    n_random_seeds: int = 20,
    n_sa_starts: int = 2,
) -> list[int]:
    """Find a good ordering of vertices via multi-restart SA.

    Strategy:
    1. Generate structured orderings (spectral, RCM, min-degree).
    2. Sample n_random_seeds random orderings and evaluate them cheaply.
    3. Pick the best n_sa_starts orderings and run SA from each.
    4. Return the globally best ordering found.

    This two-tier approach (cheap random seeding → expensive SA) ensures
    we start SA from diverse, high-quality initial points.
    """
    subgraph = graph.subgraph(vertices).copy()

    if subgraph.number_of_nodes() <= 1:
        return list(vertices)

    # Relabel to 0..n-1
    mapping = {v: i for i, v in enumerate(vertices)}
    inv_mapping = {i: v for i, v in enumerate(vertices)}
    G_int = nx.relabel_nodes(subgraph, mapping)
    n = G_int.number_of_nodes()

    adj = nx.adjacency_matrix(G_int).toarray().astype(np.uint8)

    # --- Tier 1: collect and evaluate many candidate orderings cheaply ---
    scored_candidates: list[tuple[int, int, list[int]]] = []

    # Structured orderings
    for ordering in generate_initial_orderings(G_int):
        cost, bn = _full_evaluate(adj, ordering)
        scored_candidates.append((cost, bn, list(ordering)))

    # Random orderings
    rng = random.Random(seed)
    for _ in range(n_random_seeds):
        perm = list(range(n))
        rng.shuffle(perm)
        cost, bn = _full_evaluate(adj, perm)
        scored_candidates.append((cost, bn, perm))

    # Sort by cost (ascending)
    scored_candidates.sort(key=lambda x: x[0])

    global_best_lrw = scored_candidates[0][0]
    global_best_bn = scored_candidates[0][1]
    global_best_ordering = scored_candidates[0][2]

    if global_best_lrw == 0:
        return [inv_mapping[i] for i in global_best_ordering]

    # --- Tier 2: run SA from the best few starting points ---
    for c_idx in range(min(n_sa_starts, len(scored_candidates))):
        cost, bn, ordering = scored_candidates[c_idx]
        sa_seed = (seed + c_idx) if seed is not None else None
        sa_ordering, sa_lrw, sa_bn = simulated_annealing(
            G_int,
            list(ordering),
            cost,
            bn,
            T_start=T_start,
            T_min=T_min,
            alpha=alpha,
            steps_per_temp=steps_per_temp,
            seed=sa_seed,
            verbose=False,
        )
        if sa_lrw < global_best_lrw:
            global_best_lrw = sa_lrw
            global_best_ordering = sa_ordering
            global_best_bn = sa_bn

        if global_best_lrw == 0:
            break

    if verbose:
        print(f"    Intra-cluster SA: best = {global_best_lrw} "
              f"(from {len(scored_candidates)} seeds, "
              f"{min(n_sa_starts, len(scored_candidates))} SA runs)")

    return [inv_mapping[i] for i in global_best_ordering]


def order_within_cluster(
    graph: nx.Graph,
    cluster: PathCluster,
    *,
    exact_threshold: int = 6,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> list[int]:
    """Order the vertices inside a single cluster.

    * If |cluster| <= exact_threshold: exact brute-force.
    * Otherwise: SA-based solver.

    Returns a list of vertices in the chosen ordering.
    """
    verts = sorted(cluster.vertices)
    n = len(verts)

    if n <= 1:
        return verts

    if n <= exact_threshold:
        ordering = _exact_ordering(graph, verts)
        if verbose:
            print(f"    Cluster {cluster}: exact ordering (n={n})")
    else:
        ordering = _sa_ordering(graph, verts, seed=seed, verbose=verbose)
        if verbose:
            print(f"    Cluster {cluster}: SA ordering (n={n})")

    return ordering


# ═══════════════════════════════════════════════════════════════════════════
#  Section 6 — Global SA refinement (boundary-biased)
# ═══════════════════════════════════════════════════════════════════════════

def _boundary_biased_sa(
    graph: nx.Graph,
    ordering: list[int],
    current_cost: int,
    current_bn: int,
    boundary: set[int],
    *,
    T_start: float = 3.0,
    T_min: float = 0.01,
    alpha: float = 0.97,
    steps_per_temp: int = 60,
    boundary_bias: float = 0.70,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> tuple[list[int], int, int]:
    """SA refinement of the global ordering with a bias towards moving
    boundary vertices (those connecting different clusters).

    With probability boundary_bias, one of the two swap positions is
    chosen among boundary vertices.  This focuses perturbation energy
    where it matters most for the inter-cluster interface.
    """
    rng = random.Random(seed)
    adj = nx.adjacency_matrix(graph).toarray().astype(np.uint8)
    n = len(ordering)

    if n <= 1:
        return ordering, 0, 0

    # Pre-compute positions of boundary vertices in the current ordering
    boundary_list = [v for v in ordering if v in boundary]

    cur_cost = current_cost
    cur_bn = current_bn
    best_cost = cur_cost
    best_bn = cur_bn
    best_ordering = ordering[:]

    T = T_start
    total_tried = 0
    total_accepted = 0

    while T > T_min:
        for _ in range(steps_per_temp):
            total_tried += 1

            # --- Choose move type ---
            r = rng.random()
            if r < 0.50:
                # Swap move
                if boundary_list and rng.random() < boundary_bias:
                    # At least one position from boundary
                    v1 = rng.choice(boundary_list)
                    i = ordering.index(v1)
                else:
                    i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                while j == i:
                    j = rng.randint(0, n - 1)
                ordering[i], ordering[j] = ordering[j], ordering[i]
                undo = ("swap", i, j)

            elif r < 0.80:
                # Segment reversal
                i = rng.randint(0, n - 2)
                seg_len = rng.randint(2, min(n - i, 8))
                j = i + seg_len
                old_seg = ordering[i:j][:]
                ordering[i:j] = ordering[i:j][::-1]
                undo = ("reverse", i, j, old_seg)

            else:
                # Relocate
                src = rng.randint(0, n - 1)
                dst = rng.randint(0, n - 1)
                while dst == src:
                    dst = rng.randint(0, n - 1)
                old_ordering = ordering[:]
                v = ordering.pop(src)
                ordering.insert(dst, v)
                undo = ("relocate", old_ordering)

            # Evaluate
            new_cost, new_bn = _full_evaluate(adj, ordering)
            delta = new_cost - cur_cost

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                total_accepted += 1
                cur_cost = new_cost
                cur_bn = new_bn
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_bn = cur_bn
                    best_ordering = ordering[:]
                    if verbose:
                        print(
                            f"  Global SA  T={T:.4f}  new best = {best_cost} "
                            f"(step {total_tried})"
                        )
                    if best_cost == 0:
                        return best_ordering, best_cost, best_bn
            else:
                # Undo
                kind = undo[0]
                if kind == "swap":
                    _, ii, jj = undo
                    ordering[ii], ordering[jj] = ordering[jj], ordering[ii]
                elif kind == "reverse":
                    _, ii, jj, old_seg = undo
                    ordering[ii:jj] = old_seg
                elif kind == "relocate":
                    _, old_o = undo
                    ordering[:] = old_o

        T *= alpha

    if verbose:
        rate = total_accepted / max(total_tried, 1) * 100
        print(
            f"  Global SA finished: {total_tried} trials, "
            f"{rate:.1f}% accepted, best = {best_cost}"
        )

    return best_ordering, best_cost, best_bn


# ═══════════════════════════════════════════════════════════════════════════
#  Section 7 — Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def path_clustering_lrw(
    graph: nx.Graph,
    *,
    refine: bool = True,
    exact_threshold: int = 6,
    T_start_cluster: float = 4.0,
    T_min_cluster: float = 0.01,
    alpha_cluster: float = 0.97,
    steps_cluster: int = 30,
    T_start_global: float = 3.0,
    T_min_global: float = 0.01,
    alpha_global: float = 0.97,
    steps_global: int = 40,
    boundary_bias: float = 0.70,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> tuple[int, list]:
    """Approximate the linear rank-width via path-based clustering.

    Parameters
    ----------
    graph : nx.Graph
        Input graph (any node labels).
    refine : bool
        Whether to run the final global SA refinement.
    exact_threshold : int
        Maximum cluster/TSP size for exact (brute-force) solving.
    T_start_cluster, T_min_cluster, alpha_cluster, steps_cluster
        SA parameters for intra-cluster ordering.
    T_start_global, T_min_global, alpha_global, steps_global
        SA parameters for global refinement.
    boundary_bias : float
        Probability of biasing SA moves towards boundary vertices.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress information.

    Returns
    -------
    lrw : int
        Heuristic linear rank-width (= number of emitters).
    ordering : list
        Vertex ordering in original node labels.
    """
    if graph.number_of_nodes() == 0:
        return 0, []
    if graph.number_of_nodes() == 1:
        return 0, list(graph.nodes())

    # --- Relabel to integers ---
    G_int, old_to_new, new_to_old = _relabel_to_int(graph)
    n = G_int.number_of_nodes()

    if verbose:
        print(f"Graph: {n} nodes, {G_int.number_of_edges()} edges")

    # ------------------------------------------------------------------
    #  Step 1: Extract clusters via path peeling
    # ------------------------------------------------------------------
    if verbose:
        print("\n=== Step 1: Path-peeling cluster extraction ===")
    clusters = extract_clusters(G_int, verbose=verbose)

    # ------------------------------------------------------------------
    #  Fast path: if only 1 non-trivial cluster, delegate directly
    #  to the standard SA pipeline (avoids redundant multi-seed +
    #  global refinement overhead).
    # ------------------------------------------------------------------
    non_trivial = [c for c in clusters if len(c.vertices) > 1]
    if len(non_trivial) <= 1:
        if verbose:
            print("  Single-cluster graph → delegating to SA pipeline")
        # Use the core path as a structured initial ordering hint
        if non_trivial:
            cl = non_trivial[0]
            # Build initial ordering: core path first, then remaining
            core_set = set(cl.core_path)
            init_order = list(cl.core_path) + [v for v in sorted(cl.vertices) if v not in core_set]
            # Add any trivial-cluster (isolated) vertices at the end
            covered = set(init_order)
            for c in clusters:
                if c is not cl:
                    for v in c.vertices:
                        if v not in covered:
                            init_order.append(v)
                            covered.add(v)
        else:
            init_order = list(range(n))

        # Generate diverse initial orderings and pick the best
        candidates = generate_initial_orderings(G_int)
        candidates.append(init_order)
        adj = nx.adjacency_matrix(G_int).toarray().astype(np.uint8)

        best_lrw = n + 1
        best_ordering_int = init_order
        best_bn = 0
        for cand in candidates:
            c, bn = _full_evaluate(adj, cand)
            if c < best_lrw:
                best_lrw = c
                best_ordering_int = list(cand)
                best_bn = bn

        if refine and best_lrw > 0:
            best_ordering_int, best_lrw, best_bn = simulated_annealing(
                G_int, best_ordering_int, best_lrw, best_bn,
                T_start=T_start_global, T_min=T_min_global,
                alpha=alpha_global, steps_per_temp=steps_global,
                seed=seed, verbose=verbose,
            )

        original_ordering = [new_to_old[v] for v in best_ordering_int]
        if verbose:
            print(f"\nFinal lrw (height-function cost) = {best_lrw}")
        return best_lrw, original_ordering

    if verbose:
        for i, cl in enumerate(clusters):
            print(f"  Cluster {i}: {cl}")

    # ------------------------------------------------------------------
    #  Step 2: Build cluster meta-graph
    # ------------------------------------------------------------------
    if verbose:
        print("\n=== Step 2: Cluster meta-graph ===")
    weights = _compute_cluster_weights(G_int, clusters)
    cluster_graph = _build_cluster_graph(clusters, weights)

    if verbose:
        for (i, j), w in sorted(weights.items()):
            print(f"  w({i},{j}) = {w}")

    # ------------------------------------------------------------------
    #  Step 3: Order clusters (inter-cluster TSP)
    # ------------------------------------------------------------------
    if verbose:
        print("\n=== Step 3: Inter-cluster ordering ===")
    cluster_order = order_clusters(
        clusters,
        cluster_graph,
        exact_threshold=exact_threshold,
        seed=seed,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    #  Step 4: Order vertices within each cluster
    # ------------------------------------------------------------------
    if verbose:
        print("\n=== Step 4: Intra-cluster ordering ===")

    intra_orderings: dict[int, list[int]] = {}
    for idx in cluster_order:
        cl = clusters[idx]
        intra_orderings[idx] = order_within_cluster(
            G_int,
            cl,
            exact_threshold=exact_threshold,
            seed=seed,
            verbose=verbose,
            # Pass SA parameters for large clusters
        )

    # ------------------------------------------------------------------
    #  Step 5: Assemble global ordering
    # ------------------------------------------------------------------
    global_ordering: list[int] = []
    for idx in cluster_order:
        global_ordering.extend(intra_orderings[idx])

    # Evaluate the assembled ordering
    adj = nx.adjacency_matrix(G_int).toarray().astype(np.uint8)
    assembled_cost, assembled_bn = _full_evaluate(adj, global_ordering)

    if verbose:
        print(f"\n=== Assembled ordering: cost = {assembled_cost} ===")

    # ------------------------------------------------------------------
    #  Step 6: Global SA refinement (boundary-biased)
    # ------------------------------------------------------------------
    best_ordering = global_ordering
    best_cost = assembled_cost
    best_bn = assembled_bn

    if refine and best_cost > 0:
        if verbose:
            print("\n=== Step 5: Global SA refinement (boundary-biased) ===")

        boundary = _boundary_vertices(G_int, clusters)
        if verbose:
            print(f"  Boundary vertices: {len(boundary)} / {n}")

        # Single-start global SA from the assembled ordering.
        # The clustering already gives a high-quality starting point
        # so additional random restarts are unnecessary overhead.
        restart_orderings = [global_ordering[:]]

        for r_idx, r_ord in enumerate(restart_orderings):
            r_cost, r_bn = _full_evaluate(adj, r_ord)
            sa_seed = (seed + 1000 + r_idx) if seed is not None else None
            ref_ordering, ref_cost, ref_bn = _boundary_biased_sa(
                G_int,
                r_ord,
                r_cost,
                r_bn,
                boundary,
                T_start=T_start_global,
                T_min=T_min_global,
                alpha=alpha_global,
                steps_per_temp=steps_global,
                boundary_bias=boundary_bias,
                seed=sa_seed,
                verbose=(verbose and r_idx == 0),
            )
            if ref_cost < best_cost:
                best_cost = ref_cost
                best_ordering = ref_ordering
                best_bn = ref_bn
                if verbose:
                    print(f"  Restart {r_idx}: improved to {best_cost}")
            if best_cost == 0:
                break

    # --- Map back to original labels ---
    original_ordering = [new_to_old[v] for v in best_ordering]

    if verbose:
        print(f"\nFinal lrw (height-function cost) = {best_cost}")

    return best_cost, original_ordering


# ═══════════════════════════════════════════════════════════════════════════
#  Section 8 — Convenience / verification helpers
# ═══════════════════════════════════════════════════════════════════════════

def verify_ordering(graph: nx.Graph, ordering: list) -> int:
    """Independently compute the height-function cost for an ordering."""
    G_int, old_to_new, _ = _relabel_to_int(graph)
    int_ordering = [old_to_new[v] for v in ordering]
    adj = nx.adjacency_matrix(G_int).toarray().astype(np.uint8)
    cost, _ = _full_evaluate(adj, int_ordering)
    return cost


def make_caterpillar(spine_length: int, legs: list[int] | None = None) -> nx.Graph:
    """Construct a caterpillar graph.

    Parameters
    ----------
    spine_length : int
        Number of vertices on the spine (path).
    legs : list[int], optional
        legs[i] = number of pendant vertices attached to spine vertex i.
        If None, every spine vertex gets 1 leg.
    """
    G = nx.path_graph(spine_length)
    if legs is None:
        legs = [1] * spine_length
    next_node = spine_length
    for i, n_legs in enumerate(legs):
        for _ in range(n_legs):
            G.add_edge(i, next_node)
            next_node += 1
    return G


# ═══════════════════════════════════════════════════════════════════════════
#  Section 9 — Tests
# ═══════════════════════════════════════════════════════════════════════════

def _run_tests():
    """Run sanity checks on known graph families.
    """
    import time

    print("=" * 60)
    print("  Path-Clustering LRW — Test Suite")
    print("=" * 60)

    # Build a few random graphs for speed benchmarking
    rng_test = np.random.RandomState(42)
    er30 = nx.erdos_renyi_graph(30, 0.3, seed=42)
    # Ensure connected
    while not nx.is_connected(er30):
        er30 = nx.erdos_renyi_graph(30, 0.3, seed=rng_test.randint(0, 10000))

    er40 = nx.erdos_renyi_graph(40, 0.25, seed=99)
    while not nx.is_connected(er40):
        er40 = nx.erdos_renyi_graph(40, 0.25, seed=rng_test.randint(0, 10000))

    test_cases = [
        # (name, graph, known_lrw)
        ("Path P6", nx.path_graph(6), 1),
        ("Path P8", nx.path_graph(8), 1),
        ("Cycle C6", nx.cycle_graph(6), 2),
        ("Cycle C8", nx.cycle_graph(8), 2),
        ("Cycle C10", nx.cycle_graph(10), 2),
        ("Petersen", nx.petersen_graph(), 3),
        ("Star S5", nx.star_graph(5), 1),
        ("Star S8", nx.star_graph(8), 1),
        ("Caterpillar (spine=4, legs=[1,2,1,1])",
         make_caterpillar(4, [1, 2, 1, 1]), 1),
        ("Caterpillar (spine=6, uniform legs)",
         make_caterpillar(6), 1),
        ("Erdős–Rényi G(30, 0.3)", er30, None),
        ("Erdős–Rényi G(40, 0.25)", er40, None),
    ]

    all_passed = True
    total_time = 0.0

    for name, G, known_lrw in test_cases:
        t0 = time.perf_counter()
        lrw, ordering = path_clustering_lrw(
            G, seed=42, verbose=False,
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if known_lrw is not None:
            status = "✓" if lrw <= known_lrw else "✗"
            if lrw > known_lrw:
                all_passed = False
            known_str = str(known_lrw)
        else:
            status = "·"  # benchmark only, no known optimal
            known_str = "?"

        verified = verify_ordering(G, ordering)
        consistent = "ok" if verified == lrw else f"MISMATCH (verify={verified})"

        print(
            f"  {status} {name:45s}  "
            f"lrw={lrw} (known={known_str:>2s})  "
            f"verify={consistent}  "
            f"[{elapsed:.3f}s]"
        )

    print(f"\n  Total time: {total_time:.2f}s")
    print()
    if all_passed:
        print("All tests PASSED (lrw <= known optimal for every graph).")
    else:
        print("Some tests FAILED (lrw > known optimal — may need more SA iterations).")

    return all_passed


if __name__ == "__main__":
    _run_tests()
