from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

from quantum_sensing.circuit import QuantumSensingCircuit

sim_statevector = AerSimulator(method='statevector')

class QiskitQuantumSensingCircuit(QuantumSensingCircuit):
    def __init__(self, phi_signal, circuit_parameters, hamiltonian_parameters):
        super().__init__(phi_signal, circuit_parameters, hamiltonian_parameters)
        self.num_qubits = circuit_parameters['num_qubits']
        self.circuit = QuantumCircuit(self.num_qubits)
        self.gates_applied = []

    def single_body_interaction(self, theta: float, operator: str, num_qubits: int):
        for qubit in range(num_qubits):
            if operator == 'x':
                self.circuit.rx(theta, qubit)
            elif operator == 'y':
                self.circuit.ry(theta, qubit)
            else:
                self.circuit.rz(theta, qubit)

    def double_body_interaction(self, theta: float, operator: str, interaction_strengths: list[tuple]):
        for (J_ij, i, j) in interaction_strengths:
            phi = theta * J_ij
            if operator == 'x':
                self.circuit.rxx(phi, i, j)
            elif operator == 'y':
                self.circuit.ryy(phi, i, j)
            else:
                self.circuit.rzz(phi, i, j)

    def calculate_probabilities(self) -> np.ndarray:
        self.circuit.save_statevector()
        transpiled_circuit = transpile(self.circuit, sim_statevector)
        result = sim_statevector.run(transpiled_circuit).result()
        statevector = result.get_statevector(transpiled_circuit)
        return np.abs(statevector) ** 2
