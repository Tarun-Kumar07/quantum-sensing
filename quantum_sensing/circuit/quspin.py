import numpy as np

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel

from quantum_sensing.circuit import QuantumSensingCircuit

class QuspinQuantumSensingCircuit(QuantumSensingCircuit):
    def __init__(self, phi_signal,  circuit_parameters, hamiltonian_parameters):
        super().__init__(phi_signal, circuit_parameters, hamiltonian_parameters)
        num_qubits = circuit_parameters['num_qubits']
        zero_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
        zero_state[0] = 1.0
        self.__state_vector = zero_state
        self.__basis = spin_basis_1d(L=num_qubits, pauli=True)

    def single_body_interaction(self, theta: float, operator: str, num_qubits: int):
        terms = [[theta * 0.5, i] for i in range(num_qubits)]
        h = hamiltonian(
            [[f"{operator}", terms]],
            [], basis=self.__basis,
            dtype=np.complex128,
            check_herm=False,
            check_symm=False)

        self.__evolve_state_vector(h)

    def double_body_interaction(self, theta: float, operator: str, interaction_strengths: list[tuple]):
        terms = [[theta * J, i, j] for J, i, j in interaction_strengths]
        h = hamiltonian(
            [[f"{operator}{operator}", terms]],
            [], basis=self.__basis,
            dtype=np.complex128,
            check_herm=False,
            check_symm=False)

        self.__evolve_state_vector(h)

    def __evolve_state_vector(self, h: hamiltonian):
        self.__state_vector = expm_multiply_parallel(-1j * h.tocsc()).dot(self.__state_vector)

    def calculate_probabilities(self) -> np.ndarray:
        return np.abs(self.__state_vector) ** 2
