from catalyst import vmap
from jax.scipy.integrate import trapezoid
from pennylane import numpy as pnp

from quantum_sensing import QuantumSensingCircuit


def _compute_wrapped_gaussian_prior(grid_vals: pnp.ndarray, delta: float, k_max: int = 5):
    """
    Computes the wrapped Gaussian prior distribution over the given grid values.
    """
    k_values = pnp.arange(-k_max, k_max + 1)
    # Broadcast phi_grid and k_values to create a grid of (phi + 2*pi*k)
    # grid_vals[:, None] is (grid_size, 1)
    # k_values[None, :] is (1, 2*k_max + 1)
    # phi_shifted shape: (grid_size, 2*k_max + 1)
    phi_shifted = grid_vals[:, None] + 2 * pnp.pi * k_values[None, :]
    exponent = -(phi_shifted ** 2) / (2 * delta ** 2)
    norm = pnp.sqrt(2 * pnp.pi) * delta
    return pnp.sum(pnp.exp(exponent), axis=1) / norm

def _compute_magnetization(num_qubits: int):
    """
    Calculates the magnetization (N - 2*HammingWeight) for all 2^N computational basis states.
    This is pre-computed as it's a static property of the system size.
    """
    states = pnp.arange(2 ** num_qubits)
    # Calculate Hamming weights: count set bits for each state
    # For simplicity and Catalyst compatibility, this part might need to be
    # pre-computed or handled carefully. Here, we assume pnp can handle it
    # or it's implicitly converted/jitted correctly.
    # A more robust way might be to generate this list in Python and convert to pnp.array.
    hamming_weights = pnp.array([bin(int(x)).count('1') for x in states])
    return num_qubits - 2 * hamming_weights


class BayesianCostEvaluator:

    def __init__(self, quantum_sensing_circuit:QuantumSensingCircuit, delta: float = 0.79, phi_grid_size: int = 101):
        # Pre-compute fixed components
        self.__phi_grid = pnp.linspace(-pnp.pi, pnp.pi, phi_grid_size)
        self.__prior = _compute_wrapped_gaussian_prior(self.__phi_grid, delta)
        self.__magnetization = _compute_magnetization(quantum_sensing_circuit.get_num_qubits())

        # Vectorize the circuit
        self.__batch_probability_circuit = vmap(quantum_sensing_circuit.compute_probabilities, in_axes=(0, None))
        self.__batch_expectation_circuit = vmap(quantum_sensing_circuit.compute_expectation, in_axes=(0, None))

    def compute_cost(self, params: dict) -> float:
        weighted_mse = self.compute_mse(params) * self.__prior
        return trapezoid(weighted_mse, self.__phi_grid)

    def compute_mse(self, params):
        circuit_parameters = params['circuit_parameters']
        a = params['a']

        # all_probs shape: (phi_grid_size, 2^num_qubits)
        all_probs = self.__batch_probability_circuit(self.__phi_grid, circuit_parameters)

        # phi_est shape: (2^num_qubits,)
        phi_est = a * self.__magnetization

        # error_sq shape: (phi_grid_size, 2^num_qubits)
        # phi_est[None, :] is (1, 2^num_qubits)
        # phi_grid[:, None] is (grid_size, 1)
        error_sq = (phi_est[None, :] - self.__phi_grid[:, None]) ** 2
        # mse_values shape: (phi_grid_size,)
        mse_values = pnp.sum(error_sq * all_probs, axis=1)

        return mse_values

    def compute_expectation(self, circuit_parameters):
        expectations = self.__batch_expectation_circuit(self.__phi_grid, circuit_parameters)
        return expectations

    def get_phi_grid(self):
        return self.__phi_grid

    def get_prior(self):
        return self.__prior