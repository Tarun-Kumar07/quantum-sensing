import numpy as np
import pennylane as qml
from quantum_sensing.circuit import QuantumSensingCircuit


class PennylaneQuantumSensingCircuit(QuantumSensingCircuit):
    def __init__(self, phi_signal ,circuit_parameters, hamiltonian_parameters):
        super().__init__(phi_signal, circuit_parameters, hamiltonian_parameters)
        self.num_qubits = circuit_parameters['num_qubits']
        self.device = qml.device('lightning.qubit', wires=self.num_qubits)
        self.gates_applied = []

    def single_body_interaction(self, theta: float, operator: str, num_qubits: int):
        if operator == 'x':
            single_qubit_gate = qml.RX
        elif operator == 'y':
            single_qubit_gate = qml.RY
        else:
            single_qubit_gate = qml.RZ

        for qubit in range(num_qubits):
            parameters = {
                "phi": theta,
                "wires": qubit
            }
            self.gates_applied.append((single_qubit_gate, parameters))

    def double_body_interaction(self, theta: float, operator: str, interaction_strengths: list[tuple]):
        if operator == 'x':
            two_qubit_gate = qml.IsingXX
        elif operator == 'y':
            two_qubit_gate = qml.IsingYY
        else:
            two_qubit_gate = qml.IsingZZ

        for (J_ij, i, j) in interaction_strengths:
            parameters = {
                "phi": theta * J_ij,
                "wires": [i, j]
            }
            self.gates_applied.append((two_qubit_gate, parameters))

    def calculate_probabilities(self) -> np.ndarray:
        @qml.qnode(self.device)
        def circuit():
            for operations, parameters in self.gates_applied:
                operations(**parameters)
            return qml.probs(wires=range(self.num_qubits))

        return circuit()
