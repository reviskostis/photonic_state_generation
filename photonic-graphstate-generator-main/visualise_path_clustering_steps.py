"""
Visualise every step of path_clustering_lrw on a random 15-node graph.
Run from the photonic-graphstate-generator-main directory:

    python visualise_path_clustering_steps.py

Figures are saved to  figures/path_clustering_walkthrough/
"""

from __future__ import annotations

import os
import itertools
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from path_clustering import (
    extract_clusters,
    _compute_cluster_weights,
    _build_cluster_graph,
    _boundary_vertices,
    order_clusters,
    order_within_cluster,
)
from rank_width_sa import _relabel_to_int, _full_evaluate
from lib.tableau import stabTableau
from lib.generate_graph import GraphstateGenerator

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join("figures", "path_clustering_walkthrough")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Colour palette — Wong (2011) colorblind-safe, matches the TikZ figure
# ---------------------------------------------------------------------------
CB_BLUE   = "#0072B2"   # cbBlue
CB_ORANGE = "#D55E00"   # cbOrange
CB_GREY   = "#999999"   # cbGrey
CHARCOAL  = "#2E2E2E"   # charcoal (text / node borders)
BOX_BG    = "#F8F9F9"   # neutral box background

CLUSTER_PALETTE = [
    "#0072B2",  # cbBlue
    "#D55E00",  # cbOrange
    "#009E73",  # cbGreen
    "#CC79A7",  # cbPink
    "#56B4E9",  # cbSkyBlue
    "#E69F00",  # cbAmber
    "#F0E442",  # cbYellow
    "#999999",  # cbGrey
]


def cluster_color(i: int) -> str:
    return CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def draw_graph_base(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    node_colors: list[str],
    node_labels: dict | None = None,
    highlight_edges: list[tuple] | None = None,
    alpha_edges: float = 0.55,
    node_size: int = 600,
):
    """Draw graph on *ax* with optional edge highlighting."""
    regular_edges = [e for e in G.edges() if highlight_edges is None or e not in highlight_edges and (e[1], e[0]) not in highlight_edges]
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, ax=ax,
                           edge_color=CB_GREY, alpha=alpha_edges, width=1.4)
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, ax=ax,
                               edge_color=CB_ORANGE, alpha=0.9, width=3.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_size, linewidths=1.2,
                           edgecolors=CHARCOAL)
    lbl = node_labels if node_labels is not None else {v: str(v) for v in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=lbl, ax=ax,
                            font_size=9, font_color="white", font_weight="bold")
    ax.axis("off")


def save(fig: plt.Figure, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Build structured graph: 3 parallel paths of 10 nodes, two cross-edges
# ---------------------------------------------------------------------------
# Nodes 0-9  : top path    (line 0)
# Nodes 10-19: middle path  (line 1)
# Nodes 20-29: bottom path  (line 2)
# Cross-edges: 3 -- 13  (line0 ↔ line1) and  16 -- 26  (line1 ↔ line2)
print("Building structured 3×10 path graph …")
G_orig = nx.Graph()
G_orig.add_nodes_from(range(30))
for i in range(9):
    G_orig.add_edge(i,      i + 1)       # line 0
    G_orig.add_edge(10 + i, 10 + i + 1)  # line 1
    G_orig.add_edge(20 + i, 20 + i + 1)  # line 2
G_orig.add_edge(3, 13)   # cross-edge: line 0 ↔ line 1
G_orig.add_edge(16, 26)  # cross-edge: line 1 ↔ line 2

G_int, old_to_new, new_to_old = _relabel_to_int(G_orig)

# Fixed grid layout: rows at y = 2, 1, 0; columns at x = 0..9
pos = {}
for i in range(10):
    pos[i]      = (i * 1.5, 2.0)
    pos[10 + i] = (i * 1.5, 1.0)
    pos[20 + i] = (i * 1.5, 0.0)

n = G_int.number_of_nodes()
adj = nx.adjacency_matrix(G_int,nodelist=range(n)).toarray().astype(np.uint8)
print(f"  Nodes: {n},  Edges: {G_int.number_of_edges()}")

# ============================================================
# Figure 0 — original graph
# ============================================================
print("\n[Fig 0] Original graph")
fig, ax = plt.subplots(figsize=(14, 4))
draw_graph_base(ax, G_int, pos, node_colors=[CB_BLUE] * n, node_size=420)
# for row, label in [(2.0, "Line 0 (nodes 0–9)"), (1.0, "Line 1 (nodes 10–19)"), (0.0, "Line 2 (nodes 20–29)")]:
#     ax.text(-1.2, row, label, fontsize=9, va="center", ha="right", color=CHARCOAL)
ax.set_title("Initial graph with naive labelling",
             fontsize=13, fontweight="bold")
save(fig, "00_original_graph.pdf")

# ============================================================
# Figure 1 — Step 1: path peeling & cluster extraction
# ============================================================
print("\n[Fig 1] Step 1: Path-Peeling Cluster Extraction")
clusters = extract_clusters(G_int, verbose=True)

# Assign a colour to each cluster
node_cluster = {}
for ci, cl in enumerate(clusters):
    for v in cl.vertices:
        node_cluster[v] = ci

node_colors_clusters = [cluster_color(node_cluster[v]) for v in range(n)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Create the legend patches early so they can be used on either plot
legend_patches = [
    mpatches.Patch(color=cluster_color(i),
                   label=f"Cluster {i}  (|V|={len(cl.vertices)})")
    for i, cl in enumerate(clusters)
]

# Left: graph coloured by cluster
ax = axes[0]
draw_graph_base(ax, G_int, pos, node_colors=node_colors_clusters, node_size=420)
ax.set_title("Vertices coloured by cluster", fontsize=12)

# Right: highlight core paths
ax = axes[1]
draw_graph_base(ax, G_int, pos, node_colors=node_colors_clusters, node_size=420)
for i, cl in enumerate(clusters):
    path_edges = [(cl.core_path[j], cl.core_path[j + 1])
                  for j in range(len(cl.core_path) - 1)]
    if path_edges:
        nx.draw_networkx_edges(G_int, pos, edgelist=path_edges, ax=ax,
                               edge_color=cluster_color(i),
                               width=4.0, alpha=0.85)
ax.set_title("Core paths highlighted (per cluster colour)", fontsize=12)

# Place the legend perfectly in the top right corner of the right plot
ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.0, 1.0),
          fontsize=8, framealpha=0.85)

fig.suptitle("Step 1: Path-Peeling Cluster Extraction", fontsize=14,
             fontweight="bold")
fig.tight_layout()
save(fig, "01_cluster_extraction.pdf")

# ============================================================
# Figure 2 — Step 2: cluster meta-graph
# ============================================================
print("\n[Fig 2] Step 2: Cluster Meta-Graph")
weights = _compute_cluster_weights(G_int, clusters)
cluster_graph = _build_cluster_graph(clusters, weights)

k = len(clusters)
cg_pos = nx.spring_layout(cluster_graph, seed=SEED + 1, k=2.0)
cg_node_colors = [cluster_color(i) for i in range(k)]
cg_node_sizes = [400 + 120 * len(clusters[i].vertices) for i in range(k)]

fig, ax = plt.subplots(figsize=(7, 5))
nx.draw_networkx_nodes(cluster_graph, cg_pos, ax=ax,
                       node_color=cg_node_colors,
                       node_size=cg_node_sizes,
                       edgecolors=CHARCOAL, linewidths=1.2)
cg_labels = {i: f"C{i}\n({len(clusters[i].vertices)}v)" for i in range(k)}
nx.draw_networkx_labels(cluster_graph, cg_pos, labels=cg_labels, ax=ax,
                        font_size=8, font_color="white", font_weight="bold")

edge_labels = {(i, j): str(w) for (i, j), w in weights.items()}
edge_widths = [max(0.8, weights.get((min(u, v), max(u, v)), 1) * 0.8)
               for u, v in cluster_graph.edges()]
nx.draw_networkx_edges(cluster_graph, cg_pos, ax=ax,
                       width=edge_widths, edge_color=CHARCOAL, alpha=0.7)
nx.draw_networkx_edge_labels(cluster_graph, cg_pos, edge_labels=edge_labels,
                              ax=ax, font_size=8)

ax.set_title(
    "Step 2: Cluster Meta-Graph\n"
    "(node size ∝ cluster size; edge weight = inter-cluster edges)",
    fontsize=12, fontweight="bold"
)
ax.axis("off")
save(fig, "02_cluster_metagraph.pdf")

# ============================================================
# Figure 3 — Step 3: inter-cluster ordering (TSP)
# ============================================================
print("\n[Fig 3] Step 3: inter-cluster ordering")
cluster_order = order_clusters(clusters, cluster_graph, seed=SEED, verbose=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: meta-graph with ordering path overlaid
ax = axes[0]
ordered_edges = [(cluster_order[i], cluster_order[i + 1])
                 for i in range(len(cluster_order) - 1)]
nx.draw_networkx_nodes(cluster_graph, cg_pos, ax=ax,
                       node_color=cg_node_colors,
                       node_size=cg_node_sizes,
                       edgecolors=CHARCOAL, linewidths=1.2)
# All meta-edges grey
nx.draw_networkx_edges(cluster_graph, cg_pos, ax=ax,
                       edge_color=CB_GREY, width=1.0, alpha=0.5)
# Ordered path in orange
nx.draw_networkx_edges(cluster_graph, cg_pos, edgelist=ordered_edges, ax=ax,
                       edge_color=CB_ORANGE, width=3.5, alpha=0.9,
                       arrows=True, arrowsize=20,
                       connectionstyle="arc3,rad=0.08")
nx.draw_networkx_labels(cluster_graph, cg_pos, labels=cg_labels, ax=ax,
                        font_size=8, font_color="white", font_weight="bold")
ax.set_title("TSP ordering of clusters  (highlighted path = chosen order)", fontsize=11)
ax.axis("off")

# Right: original graph with cluster order annotated
ax = axes[1]
draw_graph_base(ax, G_int, pos, node_colors=node_colors_clusters, node_size=420)
# Overlay cluster order labels near each cluster's centroid
for rank, ci in enumerate(cluster_order):
    cl = clusters[ci]
    cx = np.mean([pos[v][0] for v in cl.vertices])
    cy = np.mean([pos[v][1] for v in cl.vertices])
    ax.text(cx, cy + 0.28, f"#{rank + 1}", fontsize=11,
            ha="center", va="center",
            color=cluster_color(ci),
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cluster_color(ci),
                      alpha=0.85))
ax.set_title("Cluster order annotated on original graph", fontsize=11)

fig.suptitle("Step 3: Inter-Cluster Ordering (TSP)", fontsize=14,
             fontweight="bold")
fig.tight_layout()
save(fig, "03_cluster_ordering.pdf")

# ============================================================
# Figure 4 — Step 4: Intra-Cluster Ordering
# ============================================================
print("\n[Fig 4] Step 4: Intra-Cluster Ordering")
intra_orderings: dict[int, list[int]] = {}
for idx in cluster_order:
    intra_orderings[idx] = order_within_cluster(
        G_int, clusters[idx], seed=SEED, verbose=True
    )

# Build global ordering after assembly
global_ordering: list[int] = []
for idx in cluster_order:
    global_ordering.extend(intra_orderings[idx])

# Overlay rank within the GLOBAL ordering on each node
global_rank = {v: i + 1 for i, v in enumerate(global_ordering)}

fig, ax = plt.subplots(figsize=(14, 4))
draw_graph_base(ax, G_int, pos, node_colors=node_colors_clusters, node_size=420)
for v in G_int.nodes():
    x, y = pos[v]
    ax.text(x, y - 0.28, f"({global_rank[v]})",
            fontsize=8, ha="center", va="center", color=CHARCOAL, fontweight="bold")

ax.set_title(
    "Step 4: Intra-Cluster Ordering\n"
    "(number in parentheses = position in assembled ordering)",
    fontsize=11, fontweight="bold"
)
save(fig, "04_intra_cluster_ordering.pdf")

# ============================================================
# Figure 5 — Assembled ordering + height function
# ============================================================
print("\n[Fig 5] Assembled ordering & height function")
assembled_cost, assembled_bn = _full_evaluate(adj, global_ordering)

temp_tableau = stabTableau.get_tableau_from_adj(GraphstateGenerator.permute_adjacency_matrix(adj,global_ordering))
heights = temp_tableau.h0[1:-1]

heights_old = []
for cut in range(n-1):
    left = global_ordering[:cut+1]
    right = global_ordering[cut+1:]
    sub = adj[np.ix_(left,right)]
    from rank_width_sa import gf2_rank
    heights_old.append(gf2_rank(sub))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: graph with vertices numbered by ordering position
ax = axes[0]
label_map = {v: f"{v}\n[{global_rank[v]}]" for v in G_int.nodes()}
draw_graph_base(ax, G_int, pos, node_colors=node_colors_clusters,
                node_labels=label_map, node_size=480)
ax.set_title(
    f"Assembled vertex ordering  (cost = {assembled_cost} emitters)\n"
    "Each node shows: vertex index  [position in the linear ordering \u03c0]",
    fontsize=11, fontweight="bold"
)

# Right: height function plot
ax = axes[1]
x_vals = list(range(1, n))
ax.bar(x_vals, heights, color=CB_BLUE, alpha=0.75, edgecolor=CHARCOAL,
       linewidth=0.8, label = 'Graph Generator function')
ax.bar(x_vals,heights_old, color=CB_GREY,alpha=0.75, edgecolor=CHARCOAL,linewidth=0.8, label = 'gf2_rank function')
ax.axhline(assembled_cost, color=CB_ORANGE, linestyle="--", linewidth=2,
           label=f"max h = {assembled_cost} (emitters)")
ax.set_xlabel("Cut position  (vertex label)", fontsize=11)
ax.set_ylabel("Cut rank  h(π, i)", fontsize=11)
ax.set_title("Height function of assembled ordering", fontsize=11,
             fontweight="bold")
ax.legend(fontsize=10)
ax.set_xticks(x_vals)
ax.set_xticklabels([str(global_ordering[i]) for i in range(n - 1)],
                   rotation=45, ha="right", fontsize=7)

fig.suptitle("Step 5 Assembled Ordering & Height Function",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "05_assembled_ordering_height.pdf")

# ============================================================
# Figure 6 — Boundary vertices (used by global SA refinement)
# ============================================================
print("\n[Fig 6] Boundary vertices for global SA refinement")
boundary = _boundary_vertices(G_int, clusters)

boundary_colors = []
for v in range(n):
    if v in boundary:
        boundary_colors.append('yellow')   # boundary vertices colored yellow
    else:
        boundary_colors.append(cluster_color(node_cluster[v]))

fig, ax = plt.subplots(figsize=(14, 5))
draw_graph_base(ax, G_int, pos, node_colors=boundary_colors, node_size=420)
legend_patches = [
    mpatches.Patch(color='yellow', label=f"Boundary vertices"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
ax.set_title(
    "Step 6 \u2013 Boundary Vertices for Global SA Refinement\n"
    "Boundary vertices are highlighted: they connect different clusters and are preferentially moved by the SA",
    fontsize=11, fontweight="bold"
)
save(fig, "06_boundary_vertices.pdf")

# ============================================================
# Figure 7 — Final result after global SA refinement
# ============================================================
print("\n[Fig 7] Final result after global SA refinement")
from path_clustering import _boundary_biased_sa

ref_ordering, ref_cost, ref_bn = _boundary_biased_sa(
    G_int,
    global_ordering[:],
    assembled_cost,
    assembled_bn,
    boundary,
    seed=SEED + 1000,
    verbose=True,
)

temp_tableau = stabTableau.get_tableau_from_adj(GraphstateGenerator.permute_adjacency_matrix(adj,ref_ordering))
final_heights = temp_tableau.h0[1:-1]

final_rank = {v: i + 1 for i, v in enumerate(ref_ordering)}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: final ordering graph
ax = axes[0]
final_colors = [cluster_color(node_cluster[v]) for v in range(n)]
label_map_final = {v: f"{v}\n[{final_rank[v]}]" for v in G_int.nodes()}
draw_graph_base(ax, G_int, pos, node_colors=final_colors,
                node_labels=label_map_final, node_size=480)
ax.set_title(
    f"Final vertex ordering  (cost = {ref_cost} emitters)\n"
    "Each node shows: vertex index [position in the linear ordering \u03c0]",
    fontsize=11, fontweight="bold"
)

# Right: final height function
ax = axes[1]
x_vals = list(range(1, n))
ax.bar(x_vals, final_heights, color="#009E73", alpha=0.75,
       edgecolor=CHARCOAL, linewidth=0.8, label="After SA refinement")
ax.bar(x_vals, heights, color=CB_BLUE, alpha=0.35,
       edgecolor="none", label="Before SA refinement")
ax.axhline(ref_cost, color=CB_ORANGE, linestyle="--", linewidth=2,
           label=f"max h = {ref_cost} (emitters)")
ax.set_xlabel("Cut position  (vertex label)", fontsize=11)
ax.set_ylabel("Cut rank  h(π, i)", fontsize=11)
ax.set_title("Height function: before vs after SA refinement",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xticks(x_vals)
ax.set_xticklabels([str(ref_ordering[i]) for i in range(n - 1)],
                   rotation=45, ha="right", fontsize=7)

fig.suptitle("Step 7: Final Ordering After Global SA Refinement",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "07_final_ordering_sa_refinement.pdf")

# ============================================================
# Summary print
# ============================================================
print(f"\n{'=' * 55}")
print(f"  Algorithm summary")
print(f"{'=' * 55}")
print(f"  Graph           : 30 nodes, {G_int.number_of_edges()} edges (3 paths of 10 + 2 cross-edges)")
print(f"  Clusters found  : {len(clusters)}")
print(f"  Cluster order   : {cluster_order}")
print(f"  Assembled cost  : {assembled_cost} emitters")
print(f"  After SA refine : {ref_cost} emitters")
print(f"{'=' * 55}")
print(f"\nAll figures saved to:  {os.path.abspath(OUT_DIR)}/")
