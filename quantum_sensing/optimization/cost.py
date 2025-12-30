import multiprocessing
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
    phi = np.asarray(phi)
    k_values = np.arange(-k_max, k_max + 1).reshape(-1, 1)

    phi_shifted = phi + 2 * np.pi * k_values
    exponent = -(phi_shifted ** 2) / (2 * delta ** 2)
    gaussian_sum = np.sum(np.exp(exponent), axis=0)

    return gaussian_sum / (np.sqrt(2 * np.pi) * delta)


def state_to_magnetization(state, num_qubits):
    hamming_weight = bin(state).count('1')
    return num_qubits - 2 * hamming_weight


def global_mse_worker(args):
    """
    We pass everything needed as arguments so we don't rely on 'self'.
    """
    phi, circuit_params, ham_params, backend, num_qubits, a = args

    # We must call the circuit creation logic directly here
    circuit = create_quantum_sensing_circuit(
        phi,
        circuit_params,
        ham_params,
        backend
    )
    probs = circuit.run_circuit()

    mse = 0.0
    for state, p in enumerate(probs):
        m = state_to_magnetization(state, num_qubits)
        phi_est = a * m
        mse += ((phi_est - phi) ** 2) * p
    return mse


class CostEvaluator:
    """Encapsulates the cost function logic."""

    def __init__(self,
                 circuit_hyperparameters: CircuitHyperParameters,
                 hamiltonian_hyperparameters: HamiltonianHyperParameters,
                 pool: multiprocessing.Pool,
                 phi_precision: int = 100):
        self.__num_qubits = circuit_hyperparameters['num_qubits']
        self.__pool = pool
        self.__circuit_backend = circuit_hyperparameters['backend']
        self.__hamiltonian_hyperparameters = hamiltonian_hyperparameters
        self.__phi_range = np.linspace(-np.pi, np.pi, phi_precision)

    def evaluate(self, encoder_parameters, decoder_parameters, a) -> float:
        prior_vals_for_all_phi = prior_wrapped_gaussian(self.__phi_range)
        _, mse_for_all_phi = self.evaluate_mse_for_all_phi(encoder_parameters, decoder_parameters, a)
        costs = mse_for_all_phi * prior_vals_for_all_phi

        return simpson(costs, self.__phi_range)

    def evaluate_mse_for_all_phi(self, encoder_parameters, decoder_parameters, a) -> tuple[np.array, np.array]:
        circuit_parameters = {
            "num_qubits": self.__num_qubits,
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": decoder_parameters,
        }

        args_for_tasks = [
            (
                phi,
                circuit_parameters,
                self.__hamiltonian_hyperparameters,
                self.__circuit_backend,
                self.__num_qubits,
                a
            )
            for phi in self.__phi_range
        ]

        mse_values = self.__pool.map(global_mse_worker, args_for_tasks)

        return self.__phi_range, np.array(mse_values)

    def __mean_square_error(self, phi, circuit_parameters, a):
        circuit = create_quantum_sensing_circuit(
            phi,
            circuit_parameters,
            self.__hamiltonian_hyperparameters,
            self.__circuit_backend
        )
        probs = circuit.run_circuit()
        mse = 0.0
        for state, p in enumerate(probs):
            m = state_to_magnetization(state, self.__num_qubits)
            phi_est = a * m
            mse += ((phi_est - phi) ** 2) * p
        return mse
