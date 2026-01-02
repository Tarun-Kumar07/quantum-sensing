import pytest

from quantum_sensing import QuantumSensingCircuit
from quantum_sensing.cost_evaluator import BayesianCostEvaluator
import pennylane.numpy as pnp
from catalyst import grad, qjit

hamiltonian_parameters = {
    "rabi_frequency": (50 * (10 ** 3)),
    "omega_m": (2.75 * (10 ** 6)),
    "mu": (14 * (10 ** 3)),
    "hamiltonian_type": "zig_zag",
}

NUM_QUBITS_TO_TEST = [2, 3, 5]
CIRCUIT_HYPERPARAMETERS_TO_TEST = [
    (0, 2),
    (2, 0),
    (1, 2)
]

@pytest.mark.parametrize("num_qubits", NUM_QUBITS_TO_TEST)
@pytest.mark.parametrize("num_encoder_blocks, num_decoder_blocks", CIRCUIT_HYPERPARAMETERS_TO_TEST)
def test_forward_and_backward_pass_with_all_parameters(num_qubits, num_encoder_blocks, num_decoder_blocks):
    circuit_hyperparameters = {
        'num_qubits': num_qubits,
        'num_encoder_blocks': num_encoder_blocks,
        'num_decoder_blocks': num_decoder_blocks,
    }
    quantum_sensing_circuit = QuantumSensingCircuit(circuit_hyperparameters, hamiltonian_parameters)
    bayesian_cost_evaluator = BayesianCostEvaluator(quantum_sensing_circuit, num_qubits, 5)
    params = {
        'circuit_parameters': pnp.zeros((num_encoder_blocks + num_decoder_blocks, 3)),
        'a': 0.1
    }

    cost = bayesian_cost_evaluator.compute_cost(params)
    assert cost is not None
    bmse = bayesian_cost_evaluator.compute_bmse(params)
    assert bmse is not None

    @qjit
    def qjit_grad(params):
        # TODO : don't know why method='fd' is needed here to avoid error
        return grad(bayesian_cost_evaluator.compute_cost, method='fd')(params)

    gradients = qjit_grad(params)
    assert gradients is not None