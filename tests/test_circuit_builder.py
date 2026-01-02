import pennylane.numpy as pnp
import pytest

from quantum_sensing import QuantumSensingCircuit

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
def test_probability_circuit(num_qubits: int, num_encoder_blocks: int, num_decoder_blocks: int):
    num_blocks = 2
    circuit_parameters = {
        'num_qubits': num_qubits,
        'num_encoder_blocks': num_blocks,
        'num_decoder_blocks': num_blocks,
    }
    compute_probability = QuantumSensingCircuit(circuit_parameters, hamiltonian_parameters).compute_probabilities
    phi = 0.1
    params = pnp.zeros((2 * num_blocks, 3))

    probs = compute_probability(phi, params)
    assert probs.shape == (2**num_qubits,)

@pytest.mark.parametrize("num_qubits", NUM_QUBITS_TO_TEST)
@pytest.mark.parametrize("num_encoder_blocks, num_decoder_blocks", CIRCUIT_HYPERPARAMETERS_TO_TEST)
def test_expectation_circuit(num_qubits: int, num_encoder_blocks, num_decoder_blocks):
    circuit_hyperparameters  = {
        'num_qubits': num_qubits,
        'num_encoder_blocks': num_encoder_blocks,
        'num_decoder_blocks': num_decoder_blocks,
    }
    compute_expectation = QuantumSensingCircuit(circuit_hyperparameters, hamiltonian_parameters).compute_expectation
    phi = 0.1
    params = pnp.zeros((num_encoder_blocks + num_decoder_blocks, 3))

    expectation = compute_expectation(phi, params)
    assert expectation is not None

def test_circuit_when_parameters_are_wrong():
   circuit_hyperparameters  = {
         'num_qubits': 2,
         'num_encoder_blocks': 1,
         'num_decoder_blocks': 2,
   }
   parameters_with_wrong_shape = pnp.zeros((2, 3))
   quantum_sensing_circuit = QuantumSensingCircuit(circuit_hyperparameters, hamiltonian_parameters)

   with pytest.raises(ValueError):
       quantum_sensing_circuit.compute_probabilities(0.1, parameters_with_wrong_shape)
   with pytest.raises(ValueError):
       quantum_sensing_circuit.compute_expectation(0.1, parameters_with_wrong_shape)
