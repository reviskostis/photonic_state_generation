from typing import List, Dict, Tuple, Set, Optional, Any
import networkx as nx
import numpy as np

Vertex = int
Graph = nx.Graph
Matrix = np.array



def biadjacency_matrix(G: Graph, X: Set[Vertex]) -> Matrix:
    """Compute the biadjacency matrix between X and its complementary X^c

    Args:
        G (Graph): graph
        X (Set[Vertex]): vertex subset

    Returns:
        Matrix: output matrix
    """    
    X_comp = list(set(G.nodes()) - set(X))
    if X == [] or X_comp == []:  # empty or full subset -> rank 0
        return None

    M = np.zeros((len(X), len(X_comp)), dtype=np.uint8)
    for i, u in enumerate(X):
        for j, v in enumerate(X_comp):
            if G.has_edge(u, v):
                M[i, j] = 1
    return M

def cut_rank(G: Graph, X: Set[int]) -> int:
    """
    Compute the cut-rank of G w.r.t. the subset X in V(G).
    That is, rank over GF(2) of the adjacency-matrix between X and V\X.
    """
    M = biadjacency_matrix(G, X)
    if M is None:
        return 0
    return rank_mod2(M)

def rank_mod2(M: Matrix) -> int:
    """Compute the rank of a binary matrix (in mod 2) 

    Args:
        M (Matrix): input matrix

    Returns:
        int: rank of this matrix
    """
    M = M.copy() % 2
    n_rows, n_cols = M.shape
    rank = 0
    row = 0
    col = 0

    while row < n_rows and col < n_cols:
        # Find pivot in current column
        pivot = None
        for r in range(row, n_rows):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        # Swap pivot row to current row
        if pivot != row:
            M[[row, pivot], :] = M[[pivot, row], :]
        # Eliminate below
        for r in range(row + 1, n_rows):
            if M[r, col] == 1:
                M[r, :] ^= M[row, :]
        rank += 1
        row += 1
        col += 1

    return rank


