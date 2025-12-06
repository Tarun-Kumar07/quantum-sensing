import numpy as np
from functools import reduce

from scipy.sparse.linalg import expm_multiply
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

from quantum_sensing.circuit import QuantumSensingCircuit

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rx(theta):
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * X


def ry(theta):
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Y


def rz(theta):
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Z


def kron_n(mat, n):
    return reduce(np.kron, [mat] * n)


def rx_all(theta, n):
    return kron_n(rx(theta), n)


def ry_all(theta, n):
    return kron_n(ry(theta), n)


def rz_all(theta, n):
    return kron_n(rz(theta), n)


class QuspinQuantumSensingCircuit(QuantumSensingCircuit):
    def __init__(self, phi_signal,  circuit_parameters, hamiltonian_parameters):
        super().__init__(phi_signal, circuit_parameters, hamiltonian_parameters)
        num_qubits = circuit_parameters['num_qubits']
        zero_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
        zero_state[0] = 1.0
        self.__state_vector = zero_state
        self.__basis = spin_basis_1d(L=num_qubits, pauli=True)

    def single_body_interaction(self, theta: float, operator: str, num_qubits: int):
        rotation_matrix = None
        if operator == 'x':
            rotation_matrix = rx_all(theta, num_qubits)
        elif operator == 'y':
            rotation_matrix = ry_all(theta, num_qubits)
        elif operator == 'z':
            rotation_matrix = rz_all(theta, num_qubits)

        self.__state_vector = rotation_matrix @ self.__state_vector

    def double_body_interaction(self, theta: float, operator: str, interaction_strengths: list[tuple]):
        terms = [[theta * J, i, j] for J, i, j in interaction_strengths]
        h = hamiltonian(
            [[f"{operator}{operator}", terms]],
            [], basis=self.__basis,
            dtype=np.complex128,
            check_herm=False,
            check_symm=False)
        self.__state_vector = expm_multiply(-1j * h.tocsc(), self.__state_vector)

    def calculate_probabilities(self) -> dict:
        return np.abs(self.__state_vector) ** 2