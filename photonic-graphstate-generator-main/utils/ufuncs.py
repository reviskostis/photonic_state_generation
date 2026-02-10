from lib.tableau import *
import numpy as np
import stim
import sys
import os
import networkx as nx
import time
from typing import Any


parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)


# These function are usually part of class instances but implemented here as a standalone version


def tableau_from_Paulistrings(pauliStrs):
    num_Str = len(pauliStrs)
    new_tableau = np.zeros((num_Str, 2*num_Str), dtype=bool)
    signVector = np.zeros((num_Str), dtype=bool)
    for i in range(num_Str):
        new_tableau[i] = np.concatenate(pauliStrs[i].to_numpy())
        if pauliStrs[i].sign == -1:
            signVector[i] = True
        elif pauliStrs[i].sign == 1:
            signVector[i] = False

    return new_tableau, signVector


def get_PauliStrings(tableau, signVector=[]):
    """This method can be used as an interface to the stim library

    Keyword Arguments:
        Tableau -- optional Tableau in Pauli representation, used instead of self (default: {None})

    Returns:
        A list of stim.PauliString's equivalent to the parsed tableau
    """
    rows, columns = np.shape(tableau)
    if len(signVector) == 0:
        signVector = np.zeros(rows, dtype=bool)

    pauliArray = np.array(
        ['' for n in range(columns)], dtype=object)
    for i in range(rows):
        for j in range(columns):
            pauliArray[i] += tableau[i, j]

        if signVector[i] == True:
            pauliArray[i] = stim.PauliString("-"+pauliArray[i])
        else:
            pauliArray[i] = stim.PauliString(pauliArray[i])

    return pauliArray


def echelon_transform(pauliStrs):

    pauliStrings = pauliStrs
    tableau, _ = tableau_from_Paulistrings(pauliStrs)

    rows, columns = np.shape(tableau)
    R = np.eye(rows, dtype=bool)

    assert columns == 2 * \
        rows, f"mismatching tableau dimensions, should be n,2n but is {rows,columns}"

    nz_row_resort_ind = np.concatenate((np.where(~np.all(tableau == 0, axis=1))[
                                       0], np.where(np.all(tableau == 0, axis=1))[0]))
    tableau = tableau[nz_row_resort_ind, :]

    # transform tableau into an alternating form [[X1, Z1, X2, ...],...] dtype=bool
    alt_tab = tableau[:, np.reshape(
        [np.arange(rows), np.arange(rows) + rows], (2 * rows, 1), order='F')]
    nz_row_num = np.sum(~np.all(tableau == 0, axis=1))

    for i in range(nz_row_num):
        non_trivial_ind = np.argwhere(alt_tab[i:nz_row_num, :] == 1)
        if non_trivial_ind[0].size > 0:
            rowpivot, columnpivot = non_trivial_ind[0][0], non_trivial_ind[0][1]
            rowpivot = rowpivot+i
            if columnpivot.size == 0:
                print('Warning: Encountered empty case!')
                break
            else:
                if rowpivot != i:
                    alt_tab[[i, rowpivot],
                            :] = alt_tab[[rowpivot, i], :]

                    pauliStrings[[i, rowpivot]
                                 ] = pauliStrings[[rowpivot, i]]

                del_ind = np.argwhere(
                    alt_tab[(i + 1):nz_row_num, columnpivot] == 1)[:, 0] + i + 1
                if del_ind.size > 0:
                    alt_tab[del_ind, :] = np.mod(
                        alt_tab[del_ind, :].astype("int32") + alt_tab[i, :].astype("int32"), 2)

                    for ind in del_ind:
                        pauliStrings[ind] = pauliStrings[i] * \
                            pauliStrings[ind]

    alt_tab = alt_tab.reshape(rows, 2*rows)
    ind = np.argmax(alt_tab != 0, axis=1)
    alt_tab = alt_tab[np.argsort(ind)]
    pauliStrings = pauliStrings[np.argsort(ind)]

    tableau, _ = tableau_from_Paulistrings(pauliStrs)

    return tableau, R, pauliStrings


def heightfunction(tableau, n_p):

    rows, columns = np.shape(tableau)

    if columns != 2*rows:
        raise ValueError('The size of Tableau is improper!')

    h = np.zeros(n_p+1)

    reversed_ind = np.array(np.arange(rows, 0, -1), dtype=int)-1
    reversed_Tableau = tableau[:, np.concatenate((
        reversed_ind, rows + reversed_ind))]
    rev_tab_echelon, _, pauliStr = echelon_transform(
        pauliStrs=get_PauliStrings(stabTableau.tableau_to_pauli(reversed_Tableau)))
    B = tableau_to_bigram(rev_tab_echelon)

    for i in range(n_p+1):
        h[i] = i - np.count_nonzero(B[:, 0] >= (rows-i+1))

    return h


def tableau_to_bigram(tableau):
    """Bigram-array of given tableau

    Arguments:
        tableau -- An array of a tableau in Tableau representation

    Returns:
        Bigram combinations of the tableau
    """
    rows, columns = tableau.shape
    n = np.count_nonzero(np.all(tableau == 0, axis=1) == False)
    Bigram = np.zeros((n, 2), dtype=int)

    for i_r in range(1, n+1):
        row = tableau[i_r-1, :]
        ind = np.where(row[0:rows] + row[rows:(2 * rows)] != 0)[0]
        if len(ind):
            Bigram[i_r-1, 0] = ind[0] + 1
            Bigram[i_r-1, 1] = ind[-1] + 1
    return Bigram
