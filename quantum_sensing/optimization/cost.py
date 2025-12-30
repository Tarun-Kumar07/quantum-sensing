import multiprocessing
from typing import TypedDict

import numpy as np
from jax import Array
from scipy.integrate import simpson

import jax
import jax.numpy as jnp
import pennylane as qml

from quantum_sensing.circuit.hamiltonian_interaction_strength import interaction_strength


def create_jax_qnode(num_qubits, hamiltonian_hyperparameters):
    """
    Factory function that returns a JAX-compatible QNode.
    """
    dev = qml.device("lightning.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="jax")
    def qnode_fn(phi, encoder_params, decoder_params):
        """
        Internal circuit logic that mirrors your QuantumSensingCircuit.
        """
        # 1. Setup interactions (Pre-calculated for the QNode)
        qubit_pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        interaction_data = [(interaction_strength(i, j, hamiltonian_hyperparameters), i, j) for i, j in qubit_pairs]

        # 2. Initial State Preparation
        for i in range(num_qubits):
            qml.RY(jnp.pi / 2, wires=i)

        # 3. Helper for Body Interactions (Shared by Encoder/Decoder)
        def body_layer(params_block):
            s_rot, xx_rot, zz_rot = params_block
            for i in range(num_qubits):
                qml.RX(s_rot, wires=i)
            for J_ij, i, j in interaction_data:
                qml.IsingXX(xx_rot * J_ij, wires=[i, j])
                qml.IsingZZ(zz_rot * J_ij, wires=[i, j])

        # 4. Apply Encoder Blocks
        for block in encoder_params:
            body_layer(block)

        # 5. Sensing Layer (The phi signal)
        for i in range(num_qubits):
            qml.RZ(phi, wires=i)

        # 6. Apply Decoder Blocks
        for block in decoder_params:
            body_layer(block)

        # 7. Final Rotation and Measurement
        for i in range(num_qubits):
            qml.RX(jnp.pi / 2, wires=i)

        return qml.probs(wires=range(num_qubits))

    return qnode_fn


class CircuitHyperParameters(TypedDict):
    num_qubits: int
    num_blocks: int
    backend: str


class HamiltonianHyperParameters(TypedDict):
    hamiltonian_type: str
    rabi_frequency: float
    omega_m: float
    mu: float


def prior_wrapped_gaussian(phi_signal, delta=0.3, k_max=5):
    """
    Vectorized JAX implementation of the wrapped Gaussian prior.
    Works for both a single scalar phi and a jnp.ndarray of phis.
    """
    # 1. Ensure input is a JAX array
    phi = jnp.atleast_1d(phi_signal)

    # 2. Create the k shifts: shape (2*k_max + 1,)
    k_values = jnp.arange(-k_max, k_max + 1)

    # 3. Use broadcasting to calculate all (phi + 2*pi*k) combinations
    # phi[:, None] is (N, 1), k_values[None, :] is (1, K)
    # Resulting grid is (N, K)
    phi_shifted = phi[:, jnp.newaxis] + 2 * jnp.pi * k_values[jnp.newaxis, :]

    # 4. Calculate Gaussian components
    exponent = -(phi_shifted ** 2) / (2 * delta ** 2)
    gaussian_components = jnp.exp(exponent)

    # 5. Sum over the k-axis (axis 1) and normalize
    # This reduces the (N, K) matrix back to (N,)
    normalization = jnp.sqrt(2 * jnp.pi) * delta
    prior_array = jnp.sum(gaussian_components, axis=1) / normalization

    return prior_array


def state_to_magnetization(state, num_qubits):
    hamming_weight = bin(state).count('1')
    return num_qubits - 2 * hamming_weight


class CostEvaluator:
    """Encapsulates the cost function logic."""

    def __init__(self,
                 circuit_hyperparameters: CircuitHyperParameters,
                 hamiltonian_hyperparameters: HamiltonianHyperParameters,
                 phi_precision: int = 100):
        self.__num_qubits = circuit_hyperparameters['num_qubits']
        self.__circuit_backend = circuit_hyperparameters['backend']
        self.__hamiltonian_hyperparameters = hamiltonian_hyperparameters
        self.__phi_range = jnp.linspace(-np.pi, np.pi, phi_precision)

        # 1. Get the QNode directly from our factory
        sensing_qnode = create_jax_qnode(
            num_qubits=self.__num_qubits,
            hamiltonian_hyperparameters=self.__hamiltonian_hyperparameters,
        )

        self.__vectorized_circuit = jax.vmap(sensing_qnode, in_axes=(0, None, None))

    def evaluate(self, encoder_parameters, decoder_parameters, a) -> Array:
        prior_vals_for_all_phi = prior_wrapped_gaussian(self.__phi_range)
        _, mse_for_all_phi = self.evaluate_mse_for_all_phi(encoder_parameters, decoder_parameters, a)
        costs = mse_for_all_phi * prior_vals_for_all_phi

        return jax.scipy.integrate.trapezoid(costs, self.__phi_range)

    def evaluate_mse_for_all_phi(self, encoder_parameters, decoder_parameters, a):
        all_probs = self.__vectorized_circuit(self.__phi_range, encoder_parameters, decoder_parameters)

        # 4. Vectorized MSE Calculation (Same as before)
        states = jnp.arange(2 ** self.__num_qubits)
        magnetization_values = (2.0 * states / (2 ** self.__num_qubits - 1)) - 1.0
        phi_est = a * magnetization_values

        errors_sq = (phi_est[jnp.newaxis, :] - self.__phi_range[:, jnp.newaxis]) ** 2
        mse_values = jnp.sum(errors_sq * all_probs, axis=1)

        return self.__phi_range, mse_values
