import numpy as np
import networkx as nx
from lib.circuitSolver import *
from lib.generate_graph import *
from lib.tableau import *
from lib.LC_edge_reduction import *
from utils.ufuncs import *


# 0.99,'iterations':1700}
def run_optimization(input_graph: nx.Graph, edge_reduction=True, minLA=True, num_LA=10, opt_CNOT=True, minLA_sett={'temperature': 2000, 'cooling_rate': 0.6, 'iterations': 6000}):
    """This function can use all optimizations and takes care of the input and output of all the different functions

    Arguments:
        input_graph -- networkx graph that should be generated

    Keyword Arguments:
        edge_reduction -- if True local complementations are applied to try to reduce the total number of edges in the graph (default: {True})
        minLA -- if True applies the heuristic emission ordering on the graph (default: {True})
        num_LA -- defines how many minLA orderings are produced and takes an ordering with the least number of emitters (default: {10})
        opt_CNOT -- if True the algorithm performs a search for the best emitter for each node aiming to reduce the number of CNOTs (default: {True})

    Returns:
        an networkx graph, a stim circuit and statistics of the circuit (including #emitters, #emitter-CNOTs, #gates-on-emitter,emission order,local complementations performed)
    """
    graph = input_graph.copy()
    LC_sequence = []
    if edge_reduction:
        LCR = LCReduction(graph)
        LCR.edge_reduction(with_increase=True)
        LC_sequence = LCR.get_path()
        LC_graph = graph.copy()

        for LC_node in LC_sequence:
            LC_graph = LCReduction.local_complementation(LC_graph, LC_node)

        if LC_graph.number_of_edges() < graph.number_of_edges():
            graph = LC_graph.copy()

    emission_order = list((range(graph.number_of_nodes())))

    if minLA:
        best_ordering = []
        order_list = []
        prev_n_e = np.inf
        for i in range(num_LA):
            # np.random.shuffle(emission_order)
            emission_order, _ = GraphstateGenerator.minLa_sim_anneal(
                graph, initial_arrangement=emission_order, temperature=minLA_sett['temperature'], cooling_rate=minLA_sett['cooling_rate'], iterations=minLA_sett['iterations'])
            order_list.append(emission_order)
        for order in order_list:
            adj_matrix = nx.adjacency_matrix(
                graph.copy(), nodelist=order).toarray()
            tableau = GraphstateGenerator.get_tableau_from_adj(adj_matrix)
            h = heightfunction(tableau, len(order))

            if max(h) < prev_n_e:
                best_ordering = order
                prev_n_e = max(h)
        emission_order = best_ordering

    adj_matrix = nx.adjacency_matrix(
        graph, nodelist=emission_order).toarray()
    graph = nx.Graph(adj_matrix)

    tableau = stabTableau(GraphstateGenerator.get_tableau_from_adj(adj_matrix))
    inv_operations, num_em_cnot = algorithm(
        tableau, optimize=opt_CNOT, return_num_cnots=True)
    circuit, num_gates = generationSequence(
        inv_operations, return_n_unitaries=True)

    # check_circuit(circuit, pauliStrings=tableau.pauliStrings0)
    num_em = circuit.num_qubits-graph.number_of_nodes()
    stats = [num_em, num_em_cnot, num_gates, emission_order, np.array(emission_order)[
        LC_sequence]]

    return graph, circuit, stats
