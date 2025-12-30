from typing import TypedDict

import numpy as np
from scipy.integrate import simpson

from quantum_sensing.circuit import create_quantum_sensing_circuit


class CircuitHyperParameters(TypedDict):
    num_qubits: int
    num_blocks: int
    backend: str


class HamiltonianHyperParameters(TypedDict):
    hamiltonian_type: str
    rabi_frequency: float
    omega_m: float
    mu: float


def prior_wrapped_gaussian(phi, delta=0.3, k_max=5):
    return sum(np.exp(-((phi + 2 * np.pi * k) ** 2) / (2 * delta ** 2)) for k in range(-k_max, k_max + 1)) / (
            np.sqrt(2 * np.pi) * delta)


def state_to_magnetization(state, num_qubits):
    hamming_weight = bin(state).count('1')
    return num_qubits - 2 * hamming_weight


def mean_square_error(phi, probabilities, a):
    mse = 0.0
    num_qubits = int(np.log2(len(probabilities)))
    for state, p in enumerate(probabilities):
        m = state_to_magnetization(state, num_qubits)
        phi_est = a * m
        mse += ((phi_est - phi) ** 2) * p
    return mse


class CostEvaluator:
    """Encapsulates the cost function logic."""

    def __init__(self,
                 circuit_hyperparameters: CircuitHyperParameters,
                 hamiltonian_hyperparameters: HamiltonianHyperParameters,
                 phi_precision: int = 100):
        self.__num_qubits = circuit_hyperparameters['num_qubits']
        self.__circuit_backend = circuit_hyperparameters['backend']
        self.__hamiltonian_hyperparameters = hamiltonian_hyperparameters
        self.__phi_range = np.linspace(-np.pi, np.pi, phi_precision)

    def evaluate(self, encoder_parameters, decoder_parameters, a) -> float:
        circuit_parameters = {
            "num_qubits": self.__num_qubits,
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": decoder_parameters,
        }

        costs = []
        for phi in self.__phi_range:
            circuit = create_quantum_sensing_circuit(
                phi,
                circuit_parameters,
                self.__hamiltonian_hyperparameters,
                self.__circuit_backend
            )
            probabilities = circuit.run_circuit()
            costs.append(mean_square_error(phi, probabilities, a) * prior_wrapped_gaussian(phi))

        return simpson(costs, self.__phi_range)

    def evaluate_mse_for_all_phi(self, encoder_parameters, decoder_parameters, a) -> tuple[np.array, np.array]:
        circuit_parameters = {
            "num_qubits": self.__num_qubits,
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": decoder_parameters,
        }

        mse_values = []
        for phi in self.__phi_range:
            circuit = create_quantum_sensing_circuit(
                phi,
                circuit_parameters,
                self.__hamiltonian_hyperparameters,
                self.__circuit_backend
            )
            probs = circuit.run_circuit()
            mse_values.append(mean_square_error(phi, probs, a))

        return self.__phi_range, np.array(mse_values)