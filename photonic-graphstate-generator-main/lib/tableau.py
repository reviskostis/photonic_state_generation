import numpy as np
import stim
import sys
import os
import networkx as nx
import time
from typing import Any

# get the parent directory of the folder containing the Python file
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)


class stabTableau(object):
    """a class with functions for the graph tableau"""

    def __init__(self, tableau: list[list]) -> None:
        """Any value from the input tableau (Graph) is saved in tableau0,pauliStrings0 and h0.
        determines the number of emitter needed for the state and appends them to the tableau as Zs

        The input tableau is expected with a positive signs.

        Arguments:
            tableau -- boolean nx2n list that describes the tableau in the standard form of X and Z part of the stabilizers

        Raises:
            Exception: for improper tableau sizes
        """
        self.n_p = len(tableau)
        self.tableau = tableau

        if np.shape(tableau)[1] != 2*self.n_p:
            raise Exception(
                "The shape of the input tableau doesn't match (n,2*n)")

        self.flag = "Tableau"

        # store input values
        self.tableau0 = tableau
        self.pauliStrings0 = self.get_PauliStrings(self.tableau0)

        # Calculate heightfunction and append emitters needed
        self.h0 = self.heightfunction(self.tableau0)
        self.n_e = int(max(self.h0))
        self.n = self.n_p + self.n_e

        new_tableau = np.zeros((self.n, 2*self.n), dtype=bool)
        new_tableau[:self.n_p,
                    :self.n_p] = self.tableau[:self.n_p, :self.n_p]
        new_tableau[:self.n_p, self.n:self.n +
                    self.n_p] = self.tableau[:self.n_p, self.n_p:2*self.n_p]
        new_tableau[self.n_p:(self.n_p + self.n_e), (self.n + self.n_p):(self.n + self.n_p + self.n_e)] = np.eye(self.n_e)

        self.tableau = new_tableau
        self.pauliStrings = self.get_PauliStrings(self.tableau)
        self.signVector = np.zeros(self.n, dtype=bool)

    def tableau_from_Paulistrings(self, pauliStrings=None, overwrite=False) -> np.array:
        """Converts a list of stim.PauliString to the boolean tableau shape

        Keyword Arguments:
            pauliStrings -- possible external list of stim.PauliStrings used instead of self.PauliStrings (default: {None})
            overwrite -- allows to overwrite the class tableau element with the new tableau (default: {False})

        Returns:
            the new tableau
        """
        if overwrite:
            self.pauliStrings = pauliStrings
        if pauliStrings is None:
            pauliStrs = self.pauliStrings
        else:
            pauliStrs = pauliStrings

        num_Str = len(pauliStrs)
        new_tableau = np.zeros((num_Str, 2*num_Str), dtype=bool)
        for i in range(num_Str):
            new_tableau[i] = np.concatenate(pauliStrs[i].to_numpy())
            if pauliStrings is None or overwrite:
                if pauliStrs[i].sign == -1:
                    self.signVector[i] = True
                elif pauliStrs[i].sign == 1:
                    self.signVector[i] = False

        if pauliStrings is None or overwrite:
            self.tableau = new_tableau
        return new_tableau

    def echelon_transform(self, pauliStrs=None):
        """Implements the echelon transform to bring the current tableau to the 
        row reduced echelon form (RREF)

        Keyword Arguments:
            pauliStrs -- an optional external list of pauliStrings to overwrite (default: {None})

        Returns:
            tableau and pauliStrings in RREF
        """

        if pauliStrs is None:
            if self.flag != "Tableau":
                try:
                    self.tableau = stabTableau.pauli_to_tableau(self.tableau)
                    self.flag = "Tableau"
                except:
                    print("Unknown Tableau type")
            pauliStrings = self.pauliStrings
            # self.tableau_from_Paulistrings(pauliStrings=pauliStrings)
            tableau = self.tableau

        else:
            pauliStrings = pauliStrs
            tableau = self.tableau_from_Paulistrings(pauliStrings=pauliStrs)

        rows, columns = np.shape(tableau)
        R = np.eye(rows, dtype=bool)

        assert columns == 2 * \
            rows, f"mismatching tableau dimensions, should be n,2n but is {rows,columns}"

        nz_row_resort_ind = np.concatenate((np.where(~np.all(tableau == 0, axis=1))[
                                           0], np.where(np.all(tableau == 0, axis=1))[0]))
        tableau = tableau[nz_row_resort_ind, :]
        # R = R[nz_row_resort_ind, :]

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
                        # R[[i, rowpivot], :] = R[[rowpivot, i], :]
                        pauliStrings[[i, rowpivot]
                                     ] = pauliStrings[[rowpivot, i]]

                    del_ind = np.argwhere(
                        alt_tab[(i + 1):nz_row_num, columnpivot] == 1)[:, 0] + i + 1
                    if del_ind.size > 0:
                        alt_tab[del_ind, :] = np.mod(
                            alt_tab[del_ind, :].astype("int32") + alt_tab[i, :].astype("int32"), 2)
                        # R[del_ind, :] = np.mod(R[del_ind, :].astype(
                        #    "int32") + R[i, :].astype("int32"), 2)
                        for ind in del_ind:
                            pauliStrings[ind] = pauliStrings[i] * \
                                pauliStrings[ind]

        alt_tab = alt_tab.reshape(rows, 2*rows)
        ind = np.argmax(alt_tab != 0, axis=1)
        alt_tab = alt_tab[np.argsort(ind)]
        pauliStrings = pauliStrings[np.argsort(ind)]
        # R = R[np.argsort(ind)]

        tableau = alt_tab[:, np.concatenate(
            (2 * np.arange(rows), 2 * np.arange(rows) + 1))]

        if pauliStrs is None:
            self.tableau = self.tableau_from_Paulistrings(
                pauliStrings=pauliStrings)

        return tableau, R, pauliStrings

    def heightfunction(self, tableau, option=None):
        """Calculates the heightfunction of a given tableau

        Arguments:
            tableau -- stabilizer generator tableau

        Keyword Arguments:
            option -- 'pure' or None (default: {None})

        Raises:
            ValueError: false Tableau shape

        Returns:
            list of heightfunction values for all linear bipartitions
        """

        rows, columns = np.shape(tableau)

        if columns != 2*rows:
            raise ValueError('The size of Tableau is improper!')

        h = np.zeros(self.n_p+1)

        if option is None:
            reversed_ind = np.array(np.arange(rows, 0, -1), dtype=int)-1
            reversed_Tableau = tableau[:, np.concatenate((
                reversed_ind, rows + reversed_ind))]
            rev_tab_echelon, _, pauliStr = self.echelon_transform(
                pauliStrs=stabTableau.get_PauliStrings(reversed_Tableau))
            B = stabTableau.tableau_to_bigram(rev_tab_echelon)

            for i in range(self.n_p+1):
                h[i] = i - np.count_nonzero(B[:, 0] >= (rows-i+1))

        elif option == 'pure':
            tab_echelon, _, pauliStr = self.echelon_transform(
                pauliStrs=stabTableau.get_PauliStrings(tableau))
            B = stabTableau.tableau_to_bigram(tab_echelon)

            for i in range(self.n_p+1):
                h[i] = rows - i - np.count_nonzero(B[:, 0] > i)
        return h

    @staticmethod
    def apply_Clifford(pauliStrings: list[stim.PauliString], Gate: str, target: list):
        """Calculate the action of a Clifford gate onto a Tableau

        Arguments:
            pauliStrings -- a list of stim.PauliString representing the tableau
            Gate -- String with a valid gate name, or multiple gate with delimiter ;
            target -- list of target, must match the number of gates, two qubit gates must have a list at the entry

        Returns:
            list of stim.PauliString representing the new tableau
        """
        stim_Gates = ['I', 'X', 'Y', 'Z', 'H',
                      'S', 'SQRT_Z', 'SQRT_Z_DAG', 'CNOT', 'CX']
        if Gate in stim_Gates:
            Operation = stim.Tableau.from_named_gate(Gate)
            for i in range(len(pauliStrings)):
                pauliStrings[i] = pauliStrings[i].after(
                    Operation, target)
            return pauliStrings
        else:
            for i in Gate.split(';'):
                pauliStrings = stabTableau.apply_Clifford(
                    pauliStrings, i, target)
            return pauliStrings

    @staticmethod
    def get_PauliStrings(tableau):
        """This method can be used as an interface to the stim library

        Keyword Arguments:
            Tableau -- optional Tableau in Pauli representation, used instead of self (default: {None})

        Returns:
            A list of stim.PauliString's equivalent to the parsed tableau
        """
        rows, columns = np.shape(tableau)

        if columns != rows:
            try:
                tableau = stabTableau.tableau_to_pauli(tableau)
                rows, columns = np.shape(tableau)
            except:
                print("Unknown Tableau type")

        pauliArray = np.array(
            ['' for n in range(columns)], dtype=object)
        for i in range(rows):
            for j in range(columns):
                pauliArray[i] += tableau[i, j]

            pauliArray[i] = stim.PauliString(pauliArray[i])

        return pauliArray

    @staticmethod
    def get_tableau_from_adj(adjacency_matrix):

        n = len(adjacency_matrix)
        X_tab = np.eye(n, dtype=bool)
        tableau = np.hstack((X_tab, adjacency_matrix.astype(bool)), dtype=bool)

        return stabTableau(tableau)

    @staticmethod
    def get_tableau_from_graph(graph: nx.Graph):

        n = graph.number_of_nodes()
        X_tab = np.eye(n, dtype=bool)
        Z_tab = np.zeros((n, n), dtype=bool)
        for edge in graph.edges:
            Z_tab[edge[0], edge[1]] = True
            Z_tab[edge[1], edge[0]] = True
        tableau = np.hstack((X_tab, Z_tab), dtype=bool)

        return stabTableau(tableau)

    @staticmethod
    def pauli_to_tableau(tableau):
        """This funciton transforms a Tableau with strings of Pauli-Operators
        into the Tableau representation.

        Arguments:
            tableau -- An NxN array of strings containing single Operations of the Pauli-group

        Returns:
            (Nx2N) Tableau, an boolean array specifies X and Z's on each (NxN)-half
        """
        # assert self.flag != "Pauli", f"Tableau({self.flag}) is not in Pauli-form"
        n = tableau.shape[0]
        temp_tableau = np.zeros((n, 2 * n), dtype=bool)

        for i in range(n):
            for j in range(n):
                if tableau[i, j] == 'X':
                    temp_tableau[i, j] = True
                elif tableau[i, j] == 'Y':
                    temp_tableau[i, j] = True
                    temp_tableau[i, j + n] = True
                elif tableau[i, j] == 'Z':
                    temp_tableau[i, j + n] = True

        return temp_tableau

    @staticmethod
    def tableau_to_pauli(tableau):
        """This funciton transforms a Tableau representation
        into a Pauli representation.

        Arguments:
            tableau -- (Nx2N) Tableau, an boolean array specifies X and Z's on each (NxN)-half

        Returns:
            An NxN array of strings containing single Operations of the Pauli-group
        """
        n = tableau.shape[0]
        temp_tableau = np.empty((n, n), dtype=object)

        for i in range(n):
            for j in range(n):
                if tableau[i, j] and not tableau[i, j + n]:
                    temp_tableau[i, j] = 'X'
                elif tableau[i, j] and tableau[i, j + n]:
                    temp_tableau[i, j] = 'Y'
                elif not tableau[i, j] and tableau[i, j + n]:
                    temp_tableau[i, j] = 'Z'
                else:
                    temp_tableau[i, j] = 'I'

        return temp_tableau

    @staticmethod
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

    @staticmethod
    def stim_stab_generators(pauliStrings: list[stim.PauliString]):
        """This function creates a list compatible to parse into a
        stim pauli product measurement.

        Arguments:
            pauliStrings -- list of Paulistrings (for example complete set of generators)

        Returns:
            list of stim commands appending PP measurement to a circuit
        """
        output_products = np.empty_like(pauliStrings, dtype=object)
        for i in range(len(pauliStrings)):
            decomp = np.array([*pauliStrings[i]])
            pauliProducts = []
            indices_x = np.where(decomp == 1)[0]
            for j in indices_x:
                pauliProducts.append(stim.target_x(j))
                pauliProducts.append(stim.target_combiner())

            # shouldn't have any Y's at anytime
            indices_y = np.where(decomp == 2)[0]
            for j in indices_y:
                pauliProducts.append(stim.target_y(j))
                pauliProducts.append(stim.target_combiner())

            indices_z = np.where(decomp == 3)[0]
            for j in indices_z:
                pauliProducts.append(stim.target_z(j))
                pauliProducts.append(stim.target_combiner())

            del pauliProducts[-1]
            output_products[i] = pauliProducts

        return output_products
    pass
