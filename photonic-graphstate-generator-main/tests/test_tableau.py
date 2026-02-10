import numpy as np
import stim
import networkx as nx
import pytest
from lib.tableau import *
from lib.generate_graph import *


class TestTableauClass():

    def test_init(self):
        generators = np.array([[True, False, False, False, False, True, True, True],
                               [False, True, False, False,
                                True, False, True, True],
                               [False, False, True, False,
                                True, True, False, True],
                               [False, False, False, True, True, True, True, False]])
        tableau = stabTableau(generators)
        pauliStrs = [stim.PauliString('XZZZ'), stim.PauliString(
            'ZXZZ'), stim.PauliString('ZZXZ'), stim.PauliString('ZZZX')]
        assert np.all(list(tableau.pauliStrings0) == pauliStrs)

    def test_tableau_from_Paulistrings(self):
        generators = np.array([[True, False, False, False, False, True, True, True],
                               [False, True, False, False,
                                True, False, True, True],
                               [False, False, True, False,
                                True, True, False, True],
                               [False, False, False, True, True, True, True, False]])
        pauliStrs = [stim.PauliString('XZIZ'), stim.PauliString(
            'ZXZI'), stim.PauliString('IZXZ'), stim.PauliString('ZIZX')]
        tableau = stabTableau(generators)
        tableau.tableau_from_Paulistrings(pauliStrs, overwrite=True)
        new_tableau = np.array([[True, False, False, False, False, True, False, True],
                                [False, True, False, False,
                                 True, False, True, False],
                                [False, False, True, False,
                                 False, True, False, True],
                                [False, False, False, True, True, False, True, False]])

        assert np.all(tableau.tableau == new_tableau)

    @pytest.mark.parametrize("num_nodes", np.random.randint(3, 25, size=(10)))
    def test_echelon_transform(self, num_nodes):
        # Check pauli_tableau[:,0], get smallest index where pauli_tableau==I (0,1 or 2),
        # cut array into pauli_tableau[smallest_index:,1:], repeat
        random_graph = nx.watts_strogatz_graph(num_nodes, 2, 0.8)
        random_graph = GraphstateGenerator(
            random_graph.nodes(), random_graph.edges())
        random_tableau = stabTableau(random_graph.get_tableau())
        echelon_tableau, _, _ = random_tableau.echelon_transform()
        echelon_tableau = stabTableau.tableau_to_pauli(echelon_tableau)

        for i in range(len(echelon_tableau)-1):
            trivial_index = np.where(echelon_tableau[:, 0] == 'I')[0]
            smallest_index = trivial_index[0]
            assert np.all(trivial_index == np.arange(
                smallest_index, len(echelon_tableau)))
            assert smallest_index <= 2
            if i == 0:
                assert smallest_index > 0
            # check that the two leading Paulis are different
            if smallest_index == 2:
                assert echelon_tableau[:, 0][0] != echelon_tableau[:, 0][1]

            echelon_tableau = echelon_tableau[smallest_index:, 1:]

        pass
