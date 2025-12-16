from quantum_sensing.circuit import QuantumSensingCircuit
import os
import cirq
import qsimcirq
import numpy as np

simulator = cirq.Simulator()
qsim_simulator = qsimcirq.QSimSimulator({'t' : os.environ.get("OMP_NUM_THREADS", 1)})

class CirqQuantumSensingCircuit(QuantumSensingCircuit):
    def __init__(self, phi_signal, circuit_parameters, hamiltonian_parameters):
        super().__init__(phi_signal, circuit_parameters, hamiltonian_parameters)
        self.num_qubits = circuit_parameters["num_qubits"]
        self.circuit = cirq.Circuit()

    def single_body_interaction(self, theta: float, operator: str, num_qubits: int):
        for qubit in range(num_qubits):
            q = cirq.LineQubit(qubit)
            if operator == "x":
                self.circuit.append(cirq.rx(theta)(q))
            elif operator == "y":
                self.circuit.append(cirq.ry(theta)(q))
            else:
                self.circuit.append(cirq.rz(theta)(q))

    def double_body_interaction(self, theta: float, operator: str, interaction_strengths: list[tuple]):
        for (J_ij, i, j) in interaction_strengths:
            phi = theta * J_ij
            q_i = cirq.LineQubit(i)
            q_j = cirq.LineQubit(j)
            if operator == "x":
                self.circuit.append(cirq.XXPowGate(exponent=phi)(q_i, q_j))
            elif operator == "y":
                self.circuit.append(cirq.YYPowGate(exponent=phi)(q_i, q_j))
            else:
                self.circuit.append(cirq.ZZPowGate(exponent=phi)(q_i, q_j))

    def calculate_probabilities(self) -> np.ndarray:
        result = qsim_simulator.simulate(self.circuit)
        statevector = result.final_state_vector
        return np.abs(statevector) ** 2
