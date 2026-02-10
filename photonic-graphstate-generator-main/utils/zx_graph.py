import stim
import pyzx as zx
import stimcirq
from cirq.contrib.qasm_import import circuit_from_qasm


class ZXOptimization(object):

    def __init__(self, circuit: stim.Circuit) -> None:
        self.circuit = circuit
        self.circuit_wom, self.measurements = ZXOptimization._cir_before_meas(
            circuit)

    def stim_to_zx(self, stim_circuit: stim.Circuit):
        cirq_circ = stimcirq.stim_circuit_to_cirq_circuit(stim_circuit)
        self.zx_circ = zx.Circuit.from_qasm(cirq_circ.to_qasm())
        self.zx_graph = self.zx_circ.to_graph()
        return self.zx_circ

    def zx_to_stim(self, zx_circ, append_measurements: bool = True):
        qasm_obj = zx_circ.to_qasm()
        cirq_circuit = circuit_from_qasm(qasm_obj)
        # print(cirq_circuit)
        self.opt_stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(
            cirq_circuit)
        if append_measurements:
            self.opt_stim_circuit.append_from_stim_program_text(
                self.measurements)
        return self.opt_stim_circuit

    def try_zx_basic_opt(self):
        self.stim_to_zx(self.circuit_wom)
        zx_circ = zx.basic_optimization(self.zx_circ, do_swaps=False)
        self.zx_opt_circ = zx_circ.to_basic_gates()
        self.zx_opt_graph = self.zx_opt_circ.to_graph()
        return self.zx_opt_circ

    @staticmethod
    def _cir_before_meas(circuit):
        cir = str(circuit).split("\n")
        reduced_circ = ""
        excluded = ""
        # print(cir)
        last_meas = -1
        for i, line in enumerate(cir):
            # print(line,last_meas)
            if line.startswith('MR'):
                # print(line.split(' '))
                # pass
                last_meas = str(line.split(" ")[1])
                excluded += 'H' + ' ' + line.split(' ')[1] + "\n"
                excluded += line + "\n"
                reduced_circ = reduced_circ[:reduced_circ.rfind('\n')]
                reduced_circ = reduced_circ[:reduced_circ.rfind('\n')]+"\n"
                # print(reduced_circ+"\n")
                # print("_____")
                # reduced_circ= reduced_circ[:reduced_circ.rfind('\n')]
            elif "rec[-1]" == line.split(" ")[1]:
                excluded += 'H' + ' ' + line.split(' ')[2] + "\n"
                excluded += line.split(" ")[0] + " " + \
                    line.split(" ")[1] + " " + line.split(" ")[2] + "\n"
                if len(line.split(" ")) > 3:
                    reduced_circ += line.split(" ")[0] + " " + line.split(" ")[
                        3] + " " + line.split(" ")[4] + "\n"
            elif str(last_meas) in line:
                excluded += line + "\n"
            else:
                reduced_circ += line + "\n"
        return stim.Circuit(reduced_circ), excluded
