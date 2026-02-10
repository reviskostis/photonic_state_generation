import os
import pytest
from lib.circuitSolver import *
from lib.generate_graph import *
from lib.tableau import *
import stim
import networkx as nx
import numpy as np

# get the parent directory of the folder containing the Python file
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)


class TestClass:
    def test_Xrot(self):
        test = [stim.PauliString("XYZ"), stim.PauliString(
            "YZX"), stim.PauliString("ZXY")]
        assert rotate_toX(test, test[0], 0) == ([stim.PauliString(
            "XYZ"), stim.PauliString("YZX"), stim.PauliString("ZXY")], ['I'])

        test = [stim.PauliString("XYZ"), stim.PauliString(
            "YZX"), stim.PauliString("ZXY")]
        assert rotate_toX(test, test[1], 0) == ([stim.PauliString(
            "-YYZ"), stim.PauliString("XZX"), stim.PauliString("ZXY")], ['SQRT_Z_DAG'])

        test = [stim.PauliString("XYZ"), stim.PauliString(
            "YZX"), stim.PauliString("ZXY")]
        assert rotate_toX(test, test[1], 1) == ([stim.PauliString(
            "-XYZ"), stim.PauliString("YXX"), stim.PauliString("ZZY")], ['H'])

    def test_Zrot(self):
        test = [stim.PauliString("XYZ"), stim.PauliString(
            "YZX"), stim.PauliString("ZXY")]
        assert rotate_toZ(test, test[0], 0) == ([stim.PauliString(
            "ZYZ"), stim.PauliString("-YZX"), stim.PauliString("XXY")], ['H'])

        test = [stim.PauliString("XYZ"), stim.PauliString(
            "YZX"), stim.PauliString("ZXY")]
        assert rotate_toZ(test, test[1], 0) == ([stim.PauliString(
            "YYZ"), stim.PauliString("ZZX"), stim.PauliString("XXY")], ['SQRT_Z_DAG', 'H'])

        test = [stim.PauliString("XYZ"), stim.PauliString(
            "YZX"), stim.PauliString("ZXY")]
        assert rotate_toZ(test, test[1], 1) == ([stim.PauliString(
            "XYZ"), stim.PauliString("YZX"), stim.PauliString("ZXY")], ['I'])

    @pytest.mark.parametrize("num_nodes", np.random.randint(3, 25, size=(10)))
    def test_circuitSolver(self, num_nodes):
        randomgraph = nx.watts_strogatz_graph(num_nodes, 2, 0.8)
        randomgraph = GraphstateGenerator(
            randomgraph.nodes(), randomgraph.edges())
        tableau = stabTableau(randomgraph.get_tableau())

        operations = algorithm(
            tableau)
        circuit = generationSequence(operations)

        result = check_circuit(circuit, tableau.pauliStrings0, num_shots=10)

        assert np.all(result == False)

    @pytest.mark.parametrize("num_nodes", np.random.randint(3, 25, size=(10)))
    def test_algorithm(self, num_nodes):
        # Test the input Tableau turns into diag('Z') -> all qubits disentangled
        randomgraph = nx.watts_strogatz_graph(num_nodes, 2, 0.8)
        randomgraph = GraphstateGenerator(
            randomgraph.nodes(), randomgraph.edges())
        tableau = stabTableau(randomgraph.get_tableau())
        rows = len(tableau.tableau)
        operations = algorithm(
            tableau)
        assert np.all(tableau.tableau == np.concatenate(
            (np.zeros((rows, rows)), np.eye(rows)), axis=1))

    @pytest.mark.parametrize('num_nodes', np.random.randint(3, 25, size=(15)))
    def test_optimization(self, num_nodes):
        randomgraph = nx.watts_strogatz_graph(num_nodes, 2, 0.8)
        randomgraph = GraphstateGenerator(
            randomgraph.nodes(), randomgraph.edges())
        tableau = stabTableau(randomgraph.get_tableau())
        _, optimized = algorithm(tableau, optimize=True, return_num_cnots=True)

        tableau = stabTableau(randomgraph.get_tableau())
        _, not_optimized = algorithm(tableau, start_emitter=[
                                     0, 0, 0], optimize=False, return_num_cnots=True)

        tableau = stabTableau(randomgraph.get_tableau())
        _, not_optimized_reverse = algorithm(
            tableau, [-1, -1, -1], optimize=False, return_num_cnots=True)

        assert optimized <= not_optimized
        assert optimized <= not_optimized_reverse

        pass
