import numpy as np
from functools import reduce

from scipy.sparse.linalg import expm_multiply
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

from circuit.hamiltonian_parameters import J_zig_zag

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


def build_two_body_rotation(op, theta, hamiltonian_parameters, num_qubits):
    basis = spin_basis_1d(num_qubits)
    qubit_pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]

    J = lambda i, j: J_zig_zag(i, j, hamiltonian_parameters)
    terms = [[theta * J(i, j), i, j] for i, j in qubit_pairs]

    h = hamiltonian([[f"{op}{op}", terms]], [], basis=basis, dtype=np.complex128, check_herm=False, check_symm=False)
    return h.tocsc()

def run_circuit(circuit_parameters, hamiltonian_parameters):
    num_qubits = circuit_parameters["num_qubits"]
    encoder_parameters = circuit_parameters["encoder_parameters"]
    num_blocks = circuit_parameters["num_blocks"]
    if encoder_parameters.shape != (num_blocks, 2):
        raise ValueError(f"encoder_parameters must have shape {(num_blocks, 2)}, got {encoder_parameters.shape}")

    decoder_parameters = circuit_parameters["decoder_parameters"]
    if decoder_parameters.shape != (num_blocks, 2):
        raise ValueError(f"decoder_parameters must have shape {(num_blocks, 2)}, got {decoder_parameters.shape}")

    rx_all_pi_2 = rx_all(np.pi / 2, num_qubits)
    ry_all_pi_2 = ry_all(np.pi / 2, num_qubits)

    state_vector = np.zeros(2 ** num_qubits, dtype=np.complex128)
    state_vector[0] = 1.0
    state_vector = ry_all_pi_2 @ state_vector

    for single_rotation, theta in encoder_parameters:
        state_vector = rx_all(single_rotation, num_qubits) @ state_vector
        xx_interaction = build_two_body_rotation("x", theta, hamiltonian_parameters, num_qubits)
        state_vector = expm_multiply(-1j * xx_interaction, state_vector)
        zz_interaction = build_two_body_rotation("z", theta, hamiltonian_parameters, num_qubits)
        state_vector = expm_multiply(-1j * zz_interaction, state_vector)

    state_vector = rz_all(circuit_parameters["phi"], num_qubits) @ state_vector

    for single_rotation, theta in decoder_parameters:
        state_vector = rx_all(single_rotation, num_qubits) @ state_vector
        xx_interaction = build_two_body_rotation("x", theta, hamiltonian_parameters, num_qubits)
        state_vector = expm_multiply(-1j * xx_interaction, state_vector)
        zz_interaction = build_two_body_rotation("z", theta, hamiltonian_parameters, num_qubits)
        state_vector = expm_multiply(-1j * zz_interaction, state_vector)

    state_vector = rx_all_pi_2 @ state_vector
    return state_vector