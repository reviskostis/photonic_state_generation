import numpy as np
import stim
import sys
import os
import pytest
from typing import Any
from collections import deque
from lib.tableau import *
import copy

# get the parent directory of the folder containing the Python file
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)


def algorithm(tableau: stabTableau, optimize=True, return_num_cnots=False):
    """Implements the time-reversed generation algorithm from 
    "Li et al. Photonic resource state generation from a minimal number of quantum emitters. npj Quantum Inf 8, 11 (2022) https://doi.org/10.1038/s41534-022-00522-6"
    and adds a heuristic CNOT optimization.

    Arguments:
        tableau -- a class instance of stabTableau representing the stabilizer generator tableau

    Keyword Arguments:
        optimize -- if True a heuristic CNOT-optimization of the circuit is applied, with expense of computation time (default: {True})
        return_num_cnots -- changes the output to "operations + number of cnots" (default: {False})

    Returns:
        list of operations defined in supplementary of Li et al. paper
    """

    n_p = tableau.n_p
    num_qubits = tableau.n

    start_emitter = [0, 0, 0]

    # Initialize dictionaries for saving generation scheme
    list_photons = np.arange(1, n_p+1)
    gate_deque = deque()
    target_deque = deque()
    Up = dict.fromkeys(list_photons, None)
    Ue = dict.fromkeys(list_photons, None)
    Ue_target = dict.fromkeys(list_photons, None)
    W = dict.fromkeys(list_photons, None)
    W_target = dict.fromkeys(list_photons, None)
    W0 = deque()
    W0_target = deque()
    EmSite = dict.fromkeys(list_photons, None)
    MeasSite = dict.fromkeys(list_photons, None)
    num_em_cnots = 0
    # fin_meas = None
    for k in range(n_p):
        j = n_p - k
        _, _, pauliStrs = tableau.echelon_transform()
        bigram = stabTableau.tableau_to_bigram(tableau.tableau)

        h = tableau.heightfunction(tableau.tableau)
        if k == 0:
            dh = np.array([h[a]-h[a-1] for a in np.arange(1, n_p+1)])
            # fin_meas = np.argwhere(dh == -1)[0]+1
        if j > 0:
            dh = h[j] - h[j-1]
        else:
            dh = h[j]

        # check and apply time reversed measurement

        match dh:
            case 0 | 1:
                pass

            case -1:
                index = np.where(bigram[:, 0] > n_p)[0][0]
                g_b = pauliStrs[index]
                e_b = np.where([*g_b[n_p:num_qubits] +
                                g_b[num_qubits+n_p:2*num_qubits]])[0]
                if len(e_b) > 1:
                    # e_b[np.where(e_b == prev_em-n_p)[0]]
                    mu = n_p + e_b[start_emitter[0]]
                else:
                    mu = n_p+e_b[0]

                # get operations for W (transform the emitter to X so CNOT increases the heightfunction)
                match len(e_b):
                    case 1:
                        pauliStrs, gate = rotate_toX(pauliStrs, g_b, mu)
                        gate_deque.append(*gate)
                        target_deque.append(mu)
                        pass

                    case _:
                        for i_e in range(len(e_b)):
                            site = n_p + e_b[i_e]
                            pauliStrs, gate = rotate_toX(pauliStrs, g_b,  site)
                            gate_deque.append(*gate)
                            target_deque.append(site)

                        remain_e = np.setdiff1d(e_b, mu-n_p)
                        for i_e in remain_e:
                            site = n_p + i_e
                            gate = 'CNOT'
                            pauliStrs = stabTableau.apply_Clifford(
                                pauliStrs, gate, [mu, site])
                            gate_deque.append(gate)
                            target_deque.append([mu, site])
                            num_em_cnots += 1
                tableau.tableau_from_Paulistrings(pauliStrs, overwrite=True)

                # check and correct sign
                if tableau.signVector[index] == True:
                    gate = 'Z'
                    pauliStrs = stabTableau.apply_Clifford(
                        pauliStrs, gate, [mu])
                    gate_deque.append(gate)
                    target_deque.append(mu)

                W[j] = gate_deque.copy()
                gate_deque.clear()
                W_target[j] = target_deque.copy()
                target_deque.clear()

                # time-reversed measurement
                MeasSite[j] = mu
                pauliStrs = stabTableau.apply_Clifford(
                    pauliStrs, 'CNOT', [mu, j-1])

                # recover echelon-form
                _, _, pauliStrs = tableau.echelon_transform()
                bigram = stabTableau.tableau_to_bigram(tableau.tableau)

        # find the generator acting nontrivial on j-th photon and assuming has one nontrivial emitter site
        site_a = np.where(bigram[:, 0] == j)[0][-1]
        g_a = pauliStrs[site_a]

        # get the photon site into Z
        pauliStrs, gate = rotate_toZ(pauliStrs, g_a, j-1)

        for i in range(len(gate)):
            gate_deque.append(gate[i])

        Up[j] = gate_deque.copy()
        gate_deque.clear()

        # find and choose the possible emitters
        e_a = np.where([*g_a[n_p:num_qubits] +
                       g_a[num_qubits+n_p:2*num_qubits]])[0]

        # get operations for Ue (transform ga to Z on j and Z on all emitters)
        match len(e_a):
            case 1:
                eta = n_p + e_a[0]
                EmSite[j] = eta
                pauliStrs, gate = rotate_toZ(pauliStrs, g_a, eta)
                for i in range(len(gate)):
                    gate_deque.append(gate[i])
                    target_deque.append(eta)

            case _:
                for i_e in range(len(e_a)):
                    site = n_p + e_a[i_e]
                    pauliStrs, gate = rotate_toZ(pauliStrs, g_a, site)
                    g_a = pauliStrs[site_a]
                    for i in range(len(gate)):
                        gate_deque.append(gate[i])
                        target_deque.append(site)

                if optimize:  # j < fin_meas:
                    # Insert function that determines best emitter:
                    # try every combination of emitters applying
                    # the CNOTs that would be applied in the code now (max. n_e*(n_e-1) calls of CNOT)
                    # Test the resulting emitter columns on number of Paulis
                    # choose the path with least amout of Paulis
                    # if there are multiple solutions pick the one with most Z's
                    eta = n_p + \
                        e_a[_find_best_emitter(pauliStrs, n_p, site_a, j, e_a)]

                    # if j == 2:
                    #     eta = n_p + e_a[0]
                    # elif j == 1:
                    #     eta = n_p + e_a[-1]
                    EmSite[j] = eta
                else:
                    eta = n_p + e_a[start_emitter[1]]
                    EmSite[j] = eta

                e_a_remain = np.setdiff1d(e_a, eta-n_p)

                for i_e in range(len(e_a_remain)):
                    site = n_p + e_a_remain[i_e]
                    gate = 'CNOT'
                    pauliStrs = stabTableau.apply_Clifford(
                        pauliStrs, gate, [site, eta])

                    gate_deque.append(gate)
                    target_deque.append([site, eta])
                    num_em_cnots += 1

        tableau.tableau_from_Paulistrings(
            pauliStrings=pauliStrs, overwrite=True)

        # correct a -ZZ to +ZZ by flip the emitter
        if tableau.signVector[site_a] == True:
            gate = 'X'
            pauliStrs = stabTableau.apply_Clifford(pauliStrs, gate, [eta])
            gate_deque.append(gate)
            target_deque.append(eta)

        # emitting the photon from emitter eta
        pauliStrs = stabTableau.apply_Clifford(pauliStrs, 'CNOT', [eta, j-1])
        tableau.tableau_from_Paulistrings(
            pauliStrings=pauliStrs, overwrite=True)

        Ue[j] = gate_deque.copy()
        gate_deque.clear()
        Ue_target[j] = target_deque.copy()
        target_deque.clear()

        pauliStrs = elim_redundant_Z(pauliStrs, j-1, site_a)
        tableau.tableau_from_Paulistrings(
            pauliStrings=pauliStrs, overwrite=True)

    _, _, pauliStrs = tableau.echelon_transform()
    bigram = stabTableau.tableau_to_bigram(tableau.tableau)
    # find W0, operation that disentangles all emitters
    for i in np.arange(n_p, num_qubits, 1):
        g_c = pauliStrs[i]
        e_c = np.where([*g_c[n_p:num_qubits] +
                       g_c[num_qubits+n_p:2*num_qubits]])[0]
        kappa = n_p + e_c[start_emitter[2]]

        match len(e_c):

            case 1:
                pauliStrs, gate = rotate_toZ(pauliStrs, g_c, kappa)
                for gate_index in range(len(gate)):
                    W0.append(gate[gate_index])
                    W0_target.append(kappa)

            case _:
                # if there is more than one emitter rotate all to X and CNOT turns XX to XI
                for i_e in range(len(e_c)):
                    site = n_p + e_c[i_e]
                    pauliStrs, gate = rotate_toX(pauliStrs, g_c, site)
                    W0.append(*gate)
                    W0_target.append(site)
                e_c_remain = np.setdiff1d(e_c, kappa - n_p)

                for i_e in range(len(e_c_remain)):
                    site = n_p + e_c_remain[i_e]
                    gate = 'CNOT'
                    pauliStrs = stabTableau.apply_Clifford(
                        pauliStrs, gate, [kappa, site])
                    W0.append(gate)
                    W0_target.append([kappa, site])
                    num_em_cnots += 1

                gate = 'H'
                pauliStrs = stabTableau.apply_Clifford(
                    pauliStrs, gate, [kappa])
                W0.append(gate)
                W0_target.append(kappa)

        # tableau.tableau_from_Paulistrings(
        #    pauliStrings=pauliStrs, overwrite=True)

        pauliStrs = elim_redundant_Z(pauliStrs, kappa, i)

        tableau.tableau_from_Paulistrings(
            pauliStrings=pauliStrs, overwrite=True)

    # tableau.tableau_from_Paulistrings(
    #    pauliStrings=pauliStrs, overwrite=True)
    _, _, pauliStrs = tableau.echelon_transform()
    tableau.tableau_from_Paulistrings(
        pauliStrings=pauliStrs, overwrite=True)

    # correct any signs on the emitter generators
    for i_e in np.arange(n_p, num_qubits, 1):
        if tableau.signVector[i_e] == True:
            gate = 'X'
        else:
            gate = 'I'

        pauliStrs = stabTableau.apply_Clifford(pauliStrs, gate, [i_e])
        W0.append(gate)
        W0_target.append(i_e)

    if return_num_cnots:
        return [Up, Ue, Ue_target, W, W_target, W0, W0_target, MeasSite, EmSite], num_em_cnots
    return [Up, Ue, Ue_target, W, W_target, W0, W0_target, MeasSite, EmSite]


def rotate_toX(pauliStrs, g, site):
    """Helperfunction: Perform gates on the column to transform a specific generator to X on 'site'

    Arguments:
        pauliStrs -- current tableau represented as stim.PauliStrings
        g -- PauliString that should be transformed on 'site' to X
        site -- the site that should be a X after this function

    Returns:
        stim.PauliStrings for the new tableau and gates applied on qubit 'site'
    """
    if g[site] == 1:
        gate = 'I'
    elif g[site] == 2:
        gate = 'SQRT_Z_DAG'
    elif g[site] == 3:
        gate = 'H'
    else:
        print('stop')
    pauliStrs = stabTableau.apply_Clifford(pauliStrs, gate, [site])
    return pauliStrs, gate.split(';')


def rotate_toZ(pauliStrs, g, site):
    """Helperfunction: Perform gates on the column to transform a specific generator to Z on 'site'

    Arguments:
        pauliStrs -- current tableau represented as stim.PauliStrings
        g -- PauliString that should be transformed on 'site' to Z
        site -- the site that should be a Z after this function

    Returns:
        stim.PauliStrings for the new tableau and gates applied on qubit 'site'
    """
    if g[site] == 1:
        gate = 'H'
    elif g[site] == 2:
        gate = 'SQRT_Z_DAG;H'
    elif g[site] == 3:
        gate = 'I'
    else:
        print("stop")
    pauliStrs = stabTableau.apply_Clifford(pauliStrs, gate, [site])
    return pauliStrs, gate.split(';')


def elim_redundant_Z(pauliStrs, kappa, fixed):
    """Helperfunction: Eliminates the redundant Z's in a column after photon 'absorption'

    Arguments:
        pauliStrs -- the current tableau given by stim.PauliStrings
        kappa -- the strings with Z to be eliminated
        fixed -- the fixed tableau row that remains as Z

    Returns:
        tableau after the Z-column elimination process
    """
    condition = []
    for pauliStr in pauliStrs:
        if pauliStr[kappa] == 3:
            condition.append(True)
        else:
            condition.append(False)
    condition = np.array(condition)
    condition[fixed] = False
    elim = np.where(condition == True)[0]
    for stab_elim in elim:
        pauliStrs[stab_elim] = pauliStrs[fixed] * \
            pauliStrs[stab_elim]

    return pauliStrs


def generationSequence(operations, return_n_unitaries=False) -> stim.Circuit:
    """Takes the output of the generation algorithm to a stim.Circuit (could be replaced by other simulators)

    Arguments:
        operations -- List of operators as given by the algorithm

    Returns:
        stim.Circuit representing the emission scheme for the graph state
    """
    Up, Ue, Ue_target, W, W_target, W0, W0_target, MeasSite, EmSite = copy.deepcopy(
        operations)
    # sufficient but not complete dictionary of reversed operations
    opReverse = {'I': 'I', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 'H': 'H', 'SQRT_Z': 'SQRT_Z_DAG',
                 'SQRT_Z_DAG': 'SQRT_Z', 'CNOT': 'CNOT'}
    n_p = len(Up)
    n_gates = 0
    n_single_q_gates = 0
    # append the inverse Operation for j 1->n_p, following (only j==1: W0,) Emission, Up, Ue, Measurement, W
    circuit = stim.Circuit()
    for j in range(1, n_p+1):

        for i in range(len(W0)):
            if W0[-1] == 'I':
                _ = W0.pop()
                _ = W0_target.pop()
            else:
                circuit.append(opReverse[W0.pop()], W0_target.pop())
                n_gates += 1

        circuit.append('CNOT', [EmSite[j], j-1])

        for i in range(len(Up[j])):
            if Up[j][-1] == 'I':
                _ = Up[j].pop()
            else:
                circuit.append(opReverse[Up[j].pop()], [j-1])
                n_gates += 1

        for i in range(len(Ue[j])):
            if Ue[j][-1] == 'I':
                _ = Ue[j].pop()
                _ = Ue_target[j].pop()
            else:
                circuit.append(opReverse[Ue[j].pop()], Ue_target[j].pop())
                n_gates += 1

        if MeasSite[j] is not None:
            circuit.append('MR', MeasSite[j])
            circuit.append('CNOT', [stim.target_rec(-1), j-1])
            last_meas = np.where(
                np.array(list(MeasSite.values())) == MeasSite[j])[-1]+1
            if np.any(j != last_meas):
                circuit.append('H', MeasSite[j])
                n_gates += 1

        if W[j] is not None:
            for i in range(len(W[j])):
                if W[j][-1] == 'I':
                    _ = W[j].pop()
                    _ = W_target[j].pop()
                else:
                    if MeasSite[j] is None:
                        circuit.append(
                            opReverse[W[j].pop()], W_target[j].pop())
                        n_gates += 1
                    elif np.any(np.where(np.array(list(MeasSite.values())) == MeasSite[j])[-1]+1 != j):
                        circuit.append(
                            opReverse[W[j].pop()], W_target[j].pop())
                        n_gates += 1

    if return_n_unitaries == False:
        return circuit
    else:
        return circuit, n_gates


def _RUS_proba(eta: float, k: int):
    """Helperfunction to calculate the probabilities for a RUS gate

    Arguments:
        eta -- photon detection probability
        k -- maximum number of gate trials

    Returns:
        success, failure and abort probabilities
    """
    p_s, p_f, p_a = 0, 0, 0

    p_s = eta**2/2 * (1-(eta**2/2)**k)/(1-(eta**2/2))
    p_f = 1 - eta ** 2 / (2 - eta ** 2)
    p_a = 1-(p_s+p_f)

    return p_s, p_f, p_a


def _append_RUS_gate(circuit: stim.Circuit, qb_a, qb_b, p_s) -> None:
    """Helperfunction to append a RUS-CZ-gate to the circuit

    Arguments:
        circuit -- a stim.Circuit at the state where the RUS_gate should be applied
        qb_a -- choice of target-a
        qb_b -- choice of target-b
        p_s -- success probability for the gate

    Returns:
        None, changes input circuit inplace
    """
    circuit.append('H', qb_b)
    circuit.append("CZ", [qb_a, qb_b])
    p_ff = (1-p_s)/3
    phase_erasure_channel = [0, 0, p_ff,
                             0, 0, 0, 0,
                             0, 0, 0, 0,
                             p_ff, 0, 0, p_ff]

    circuit.append("PAULI_CHANNEL_2", [qb_a, qb_b], phase_erasure_channel)
    circuit.append('H', qb_b)

    pass


def generationSequenceRUS(operations, eta: float, k: int) -> stim.Circuit:
    """Using this circuit generation functions replacing all CNOT-gates between emitters with the RUS-CZ-gate
    with the same number of trials per gate and photon detection probability.

    There might be a problem with Detector sampling from this circuit which hence a higher success probability for Graph generation.
    Using the stim.TableauSimulator resolves this problem.

    Arguments:
        operations -- List of operators as given by the algorithm
        eta -- photon detection probability
        k -- number of trial per RUS-gate

    Returns:
        A stim.Circuit with possibility of spin entangling gate failures
    """
    Up, Ue, Ue_target, W, W_target, W0, W0_target, MeasSite, EmSite = copy.deepcopy(
        operations)

    # sufficient but not complete dictionary of reversed operations
    opReverse = {'I': 'I', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 'H': 'H', 'SQRT_Z': 'SQRT_Z_DAG',
                 'SQRT_Z_DAG': 'SQRT_Z', 'CNOT': 'CNOT'}
    p_s, p_f, p_a = _RUS_proba(eta, k)
    print(f'P_RUS_succes = {p_s}',
          f'P_RUS_failure = {p_f}', f'P_RUS_abort = {p_a}')
    n_p = len(Up)
    n_gates = 0
    n_single_q_gates = 0
    # append the inverse Operation for j 1->n_p, following (only j==1: W0,) Emission, Up, Ue, Measurement, W
    circuit = stim.Circuit()
    for j in range(1, n_p+1):

        for i in range(len(W0)):
            if W0[-1] == 'I':
                _ = W0.pop()
                _ = W0_target.pop()
            else:
                gate = W0.pop()
                target = W0_target.pop()
                if gate != 'CNOT':
                    circuit.append(opReverse[gate], target)
                    n_gates += 1
                else:
                    # circuit.append(opReverse[gate], target)
                    # circuit.append_operation(
                    #     "PAULI_CHANNEL_2", target, phase_erasure_channel)
                    _append_RUS_gate(circuit, target[0], target[1], p_s)
                    n_gates += 1

        circuit.append('CNOT', [EmSite[j], j-1])
        # circuit.append('CNOT', [EmSite[j], j-1+n_p+3])
        # circuit.append('H', [j-1+n_p+3])

        for i in range(len(Up[j])):
            if Up[j][-1] == 'I':
                _ = Up[j].pop()
            else:
                circuit.append(opReverse[Up[j].pop()], [j-1])
                n_gates += 1

        for i in range(len(Ue[j])):
            if Ue[j][-1] == 'I':
                _ = Ue[j].pop()
                _ = Ue_target[j].pop()
            else:
                gate = Ue[j].pop()
                target = Ue_target[j].pop()
                if gate != 'CNOT':
                    circuit.append(opReverse[gate], target)
                    n_gates += 1
                else:
                    _append_RUS_gate(circuit, target[0], target[1], p_s)
                    # circuit.append(opReverse[gate], target)
                    # circuit.append_operation(
                    #     "PAULI_CHANNEL_2", target, phase_erasure_channel)
                    n_gates += 1

        if MeasSite[j] is not None:
            circuit.append('MR', MeasSite[j])
            circuit.append('CNOT', [stim.target_rec(-1), j-1])
            last_meas = np.where(
                np.array(list(MeasSite.values())) == MeasSite[j])[-1]+1
            if np.any(j != last_meas):
                circuit.append('H', MeasSite[j])
                n_gates += 1

        if W[j] is not None:
            for i in range(len(W[j])):
                # or (W[j][0] == 'H' and W_target[j][0] == MeasSite[j]):
                if W[j][-1] == 'I':
                    _ = W[j].pop()
                    _ = W_target[j].pop()
                else:
                    if MeasSite[j] is None:
                        gate = W[j].pop()
                        target = W_target[j].pop()
                        if gate != 'CNOT':
                            circuit.append(opReverse[gate], target)
                            n_gates += 1
                        else:
                            _append_RUS_gate(
                                circuit, target[0], target[1], p_s)
                            # circuit.append(opReverse[gate], target)
                            # circuit.append_operation(
                            #     "PAULI_CHANNEL_2", target, phase_erasure_channel)
                            n_gates += 1
                    elif np.any(np.where(np.array(list(MeasSite.values())) == MeasSite[j])[-1]+1 != j):
                        gate = W[j].pop()
                        target = W_target[j].pop()
                        if gate != 'CNOT':
                            circuit.append(opReverse[gate], target)
                            n_gates += 1
                        else:
                            _append_RUS_gate(
                                circuit, target[0], target[1], p_s)
                            # circuit.append(opReverse[gate], target)
                            # circuit.append_operation(
                            #     "PAULI_CHANNEL_2", target, phase_erasure_channel)
                            n_gates += 1
    # circuit.append('H', np.arange(n_p, circuit.num_qubits, 1))
    return circuit


def _append_gate_Tableau(tableau: stim.TableauSimulator, gate, targets):
    """Helperfunction to apply the algorithm gates to the Tableau

    Arguments:
        tableau -- stim.TableauSimulator
        gate -- String of the gate to be applied
        targets -- Target('s) of the gate to be applied on
    """

    if type(targets) is not list:
        targets = [targets]
    if gate == 'H':
        tableau.h(*targets)
    elif gate == 'X':
        tableau.x(*targets)
    elif gate == 'Y':
        tableau.y(*targets)
    elif gate == 'Z':
        tableau.z(*targets)
    elif gate == 'SQRT_Z':
        tableau.s(*targets)
    elif gate == 'SQRT_Z_DAG':
        tableau.s_dag(*targets)
    elif gate == 'CNOT':
        tableau.cnot(*targets)
    pass


def _append_Tableau_RUS(tableau: stim.TableauSimulator, qb_a, qb_b, p_s, p_f, p_a, deterministic_abort=False):
    """Helperfunction to apply the action of RUS-gates with abort and failure cases

    Arguments:
        tableau -- stim.TableauSimulator track all gates applied from the algorithm
        qb_a -- Qubit A to apply gates on
        qb_b -- Qubit B to apply gate on
        p_s -- RUS-gate success probability
        p_f -- RUS-gate failure probability
        p_a -- RUS-gate abort probability

    Keyword Arguments:
        deterministic_abort -- if True the gate is aborted intentionally (default: {False})
    """
    result = ['success', 'abort', 'failure']

    action = np.random.choice(result, p=[p_s, p_a, p_f])

    if action == 'abort' or deterministic_abort:
        print(f'RUS-CNOT-gate aborted')
        pass
    elif action == 'success':
        print(f'RUS-CNOT-gate succeeded')
        tableau.cnot(qb_a, qb_b)
    else:
        error = np.random.choice(['II', 'ZI', 'IX', 'ZX'])

        if error == 'II':
            print(f'RUS-CNOT-gate failed with error IxI')
            pass
        elif error == 'ZI':
            print(f'RUS-CNOT-gate failed with error ZxI')
            tableau.z(qb_a)
        elif error == 'IX':
            print(f'RUS-CNOT-gate failed with error IxX')
            tableau.x(qb_b)
        elif error == 'ZX':
            print(f'RUS-CNOT-gate failed with error ZxX')
            tableau.z(qb_a)
            tableau.x(qb_b)

    pass


def Tableau_generationSequenceRUS(operations: list, eta: float, k: int, proba: list = None, without_noise: bool = False, number_of_cnots: int = None, deterministic_abort: list = None) -> stim.TableauSimulator:
    """Creates a stim.TableauSimulator from the inverse operation list of gates from Bikun Li's algorithm. 
    The CNOT-gates between emitters can be made probabilistic, based on RUS-gates

    Inputs:
        -operations[list]: output of Bikun Li's algorithm defining all operations of the emission circuit
        -eta[float]: 1-eta probability of photon-loss during RUS-gate execution
        -k[int]: assinged trials for one RUS-gate (all RUS-gates in circuit considered with same k)
        -proba[list: [p_s,p_f,p_a] or None]: A list containing manual values for the RUS-gate probabilities
        -withou_noise[bool]: Deterministic generation if 'True'
        -number_of_cnots[int or None]: specifies the number emitter-cnots for the determinisic abort
        -deterministic_abort[list: [False,False,True,...] or None]: Defines the i-th CNOT in the circuit to be a Id action instead of CNOT

    Returns:
        stim.TableauSimulator - the resulting tableau after (probabilistic) generation scheme
    """
    if deterministic_abort != None and len(deterministic_abort) != number_of_cnots:
        raise AssertionError(
            'The lengths of deterministic abort list must match the given number of CNOTs in the circuit')

    Up, Ue, Ue_target, W, W_target, W0, W0_target, MeasSite, EmSite = copy.deepcopy(
        operations)
    # sufficient but not complete dictionary of reversed operations
    opReverse = {'I': 'I', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 'H': 'H', 'SQRT_Z': 'SQRT_Z_DAG',
                 'SQRT_Z_DAG': 'SQRT_Z', 'CNOT': 'CNOT'}
    if without_noise:
        p_s, p_f, p_a = 1, 0, 0
    elif proba is not None:
        p_s, p_f, p_a = proba[0], proba[1], proba[2]
        assert p_s+p_f+p_a == 1
    else:
        p_s, p_f, p_a = _RUS_proba(eta, k)

    print(f'P_RUS_succes = {p_s}',
          f'P_RUS_failure = {p_f}', f'P_RUS_abort = {p_a}')
    n_p = len(Up)
    n_gates = 0
    cnot_position = 0
    # append the inverse Operation for j 1->n_p, following (only j==1: W0,) Emission, Up, Ue, Measurement, W
    tableauSim = stim.TableauSimulator()
    for j in range(1, n_p+1):

        for i in range(len(W0)):
            if W0[-1] == 'I':
                _ = W0.pop()
                _ = W0_target.pop()
            else:
                gate = W0.pop()
                target = W0_target.pop()
                if gate != 'CNOT':
                    _append_gate_Tableau(tableauSim, opReverse[gate], target)
                    n_gates += 1
                else:
                    if deterministic_abort is not None:
                        cnot_abort = deterministic_abort[cnot_position]
                        cnot_position += 1
                    else:
                        cnot_abort = False
                    _append_Tableau_RUS(
                        tableauSim, target[0], target[1], p_s, p_f, p_a, deterministic_abort=cnot_abort)
                    n_gates += 1

        _append_gate_Tableau(tableauSim, 'CNOT', [EmSite[j], j-1])
        # circuit.append('CNOT', [EmSite[j], j-1+n_p+3])
        # circuit.append('H', [j-1+n_p+3])

        for i in range(len(Up[j])):
            if Up[j][-1] == 'I':
                _ = Up[j].pop()
            else:
                gate = Up[j].pop()
                target = [j-1]
                _append_gate_Tableau(tableauSim, opReverse[gate], target)
                n_gates += 1

        for i in range(len(Ue[j])):
            if Ue[j][-1] == 'I':
                _ = Ue[j].pop()
                _ = Ue_target[j].pop()
            else:
                gate = Ue[j].pop()
                target = Ue_target[j].pop()
                if gate != 'CNOT':
                    _append_gate_Tableau(tableauSim, opReverse[gate], target)
                    n_gates += 1
                else:
                    if deterministic_abort is not None:
                        cnot_abort = deterministic_abort[cnot_position]
                        cnot_position += 1
                    else:
                        cnot_abort = False
                    _append_Tableau_RUS(
                        tableauSim, target[0], target[1], p_s, p_f, p_a, deterministic_abort=cnot_abort)
                    n_gates += 1

        if MeasSite[j] is not None:
            if tableauSim.measure(MeasSite[j]):
                tableauSim.x(j-1)

            last_meas = np.where(
                np.array(list(MeasSite.values())) == MeasSite[j])[-1]+1
            if np.any(j != last_meas):
                _append_gate_Tableau(tableauSim, 'H', MeasSite[j])
                n_gates += 1

        if W[j] is not None:
            for i in range(len(W[j])):
                # or (W[j][0] == 'H' and W_target[j][0] == MeasSite[j]):
                if W[j][-1] == 'I':
                    _ = W[j].pop()
                    _ = W_target[j].pop()
                else:
                    if MeasSite[j] is None:
                        gate = W[j].pop()
                        target = W_target[j].pop()
                        if gate != 'CNOT':
                            _append_gate_Tableau(
                                tableauSim, opReverse[gate], target)
                            n_gates += 1
                        else:
                            if deterministic_abort is not None:
                                cnot_abort = deterministic_abort[cnot_position]
                                cnot_position += 1
                            else:
                                cnot_abort = False
                            _append_Tableau_RUS(
                                tableauSim, target[0], target[1], p_s, p_f, p_a, deterministic_abort=cnot_abort)
                            n_gates += 1
                    elif np.any(np.where(np.array(list(MeasSite.values())) == MeasSite[j])[-1]+1 != j):
                        gate = W[j].pop()
                        target = W_target[j].pop()
                        if gate != 'CNOT':
                            _append_gate_Tableau(
                                tableauSim, opReverse[gate], target)
                            n_gates += 1
                        else:
                            if deterministic_abort is not None:
                                cnot_abort = deterministic_abort[cnot_position]
                                cnot_position += 1
                            else:
                                cnot_abort = False
                            _append_Tableau_RUS(
                                tableauSim, target[0], target[1], p_s, p_f, p_a, deterministic_abort=cnot_abort)
                            n_gates += 1
    return tableauSim


def check_circuit(circuit, pauliStrings, num_shots=10000) -> np.ndarray:
    """Checks if the generation circuit creates the right state on the photons. 
    Appends a measurement for the stim.PauliStrings to check and samples the success of all given PauliStrings

    Arguments:
        circuit -- a graph state generation circuit 
        pauliStrings -- the PauliStrings to check on the state (expecting the stabilizer generators of the graph state)

    Keyword Arguments:
        num_shots -- number of circuit sampling shots (default: {10000})

    Returns:
        A list with the sampled results for each PauliString
    """
    stab_generators = stabTableau.stim_stab_generators(pauliStrings)
    test_result = np.zeros(len(pauliStrings), dtype=bool)
    for i in range(len(stab_generators)):
        temp_circuit = circuit.copy()
        temp_circuit.append('MPP', stab_generators[i])
        temp_circuit.append('Detector', [stim.target_rec(-1)])
        sampler = temp_circuit.compile_detector_sampler()
        if not np.all(sampler.sample(shots=num_shots) == False):
            print("Circuit generates wrong graph")
            print(stab_generators[i])
            test_result[i] = True
            temp_circuit.diagram('timeline-svg')
    if np.all(test_result == False):
        print("Circuit generates correct graph")
    return test_result


def check_tableau(tableau: stim.TableauSimulator, pauliStrings: np.array) -> list:
    """Checks if all the canonical stabilizers of the tableau commute with each stim.PauliString
    in the pauliStrings

    Arguments:
        tableau -- a stim.TableauSimulator to check the PauliStrings
        pauliStrings -- a list of stim.PauliStrings to check

    Returns:
        The results if the stabilizers commute with the tableau
    """
    measurement_results = []
    tableau_result = tableau.canonical_stabilizers()
    for generators in pauliStrings:
        for S in tableau_result:
            if S.commutes(generators):
                measurement_results.append(False)
            else:
                measurement_results.append(True)
    if np.all(np.array(measurement_results) == False):
        print("Circuit generates correct graph")
    return measurement_results


def _find_best_emitter(pauliStrings, n_p, site_a, j, em_list) -> int:
    """Helperfunction selects an emitter to absorb the next photon into.
    The selection should reduce the stabilizer weight on the emitters 
    and aims to reduce the number of needed entangling gates between them.

    Arguments:
        pauliStrings -- current tableau in stim.PauliStrings 
        n_p -- the total number of photons in the graph
        site_a -- the index of stim.PauliString to be transformed into Z_jZ_em[i]
        j -- the number of the photon to be absorbed
        em_list -- a list of possible emitters to absorb the photon

    Returns:
        the number of the chosen emitter
    """
    num_list = np.array([], dtype=int)
    num_Zs_list = np.array([], dtype=int)
    for eta in em_list:
        pauliStr_trial = pauliStrings.copy()
        e_a_remain = np.setdiff1d(em_list, eta)
        for i_e in range(len(e_a_remain)):
            site = e_a_remain[i_e]
            pauliStr_trial = stabTableau.apply_Clifford(
                pauliStr_trial, 'CNOT', [site+n_p, eta+n_p])

        if pauliStr_trial[site_a].sign == -1+0j:
            pauliStr_trial = stabTableau.apply_Clifford(
                pauliStr_trial, 'X', [eta+n_p])

        pauliStr_trial = stabTableau.apply_Clifford(
            pauliStr_trial, 'CNOT', [eta+n_p, j-1])

        pauliStr_trial = elim_redundant_Z(pauliStr_trial, j-1, site_a)

        # empty_tab = stabTableau(
        #     np.zeros((len(pauliStr_trial), 2*len(pauliStr_trial)), dtype=bool))
        # _, _, pauliStr_trial = empty_tab.echelon_transform(
        #     pauliStrs=pauliStr_trial)

        # evaluate current emitter
        num_paulis = 0
        num_Zs = 0
        em_trial = [i[n_p:] for i in pauliStr_trial]
        # print(em_trial)
        for row in range(len(em_trial)):
            num_paulis += sum(np.array([*em_trial[row]]) > 0)
            num_Zs += sum(np.array([*em_trial[row]]) == 3)
        num_list = np.append(num_list, [num_paulis])
        num_Zs_list = np.append(num_Zs_list, [num_Zs])
    min_list = np.where(num_list == min(num_list))[0]
    max_Zs = np.where(num_Zs_list == max(num_Zs_list))[0]
    if len(min_list) == 1:
        return int(min_list[0])
    else:
        if np.intersect1d(min_list, max_Zs).size:
            return int(np.intersect1d(min_list, max_Zs)[0])
        return int(-1)


def graph_em_coloring(circuit: stim.Circuit) -> np.array:
    """graph plot utility function, to color the photons according to their emitter

    Arguments:
        circuit -- a full graph state generation circuit

    Returns:
        an array sorting the photons (nodes in the graph) to the emitter
    """
    n = circuit.num_qubits
    cir = str(circuit).split("\n")
    em_list = set()
    for line in cir:
        if line.startswith('MR'):
            em_list.add(line.split(" ")[1])
    n_p = n-len(em_list)
    emitter_ind = np.empty((n_p, 2))
    for line in cir:
        if line.startswith("CX"):
            qb = line.split(" ")[1:]
            for ind, qubit in enumerate(qb):
                if not (qubit.startswith("rec")):
                    if int(qubit) < n_p:
                        if not (qb[ind-1].startswith("rec")):
                            emitter_ind[int(qubit)] = np.array(
                                [int(qubit)+1, int(qb[ind-1]) - n_p])

    nodes_group = emitter_ind[emitter_ind[:, 1].argsort()]
    nodes_group = np.split(nodes_group[:, 0], np.unique(
        nodes_group[:, 1], return_index=True)[1][1:])

    return nodes_group
