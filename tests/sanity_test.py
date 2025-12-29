import pytest

import numpy as np
from quantum_sensing.circuit import create_quantum_sensing_circuit
from quantum_sensing.circuit.qiskit import QiskitQuantumSensingCircuit

hamiltonian_parameters = {
    "rabi_frequency": 50e3,
    "omega_m": 2.75e6,
    "mu": 10000,
    "hamiltonian_type": "com",
}

@pytest.mark.parametrize("backend", [ 'quspin', 'qiskit', 'pennylane', 'cirq' ])
def test_sanity_check(backend):
    '''
    All the parameterized rotations are set to zero in this test.
    This applies RY(pi/2) and RX(pi/2) on all qubits.
    When qubit starts with |0>, they end up at |+>, so all probabilities must be equally distributed.
    '''
    num_blocks = 2 
    num_qubits = 4
    circuit_parameters = {
        "num_qubits": num_qubits,
        "encoder_parameters": np.zeros((num_blocks, 3)),
        "decoder_parameters": np.zeros((num_blocks, 3)),
    }

    circuit = create_quantum_sensing_circuit(0, circuit_parameters, hamiltonian_parameters)
    probabilities = circuit.run_circuit()

    np.testing.assert_allclose(probabilities, 1/(2**num_qubits), rtol=1e-5, atol=1e-8)

def test_invalid_hamiltonian_type():
    num_blocks = 1
    num_qubits = 2
    circuit_parameters = {
        "num_qubits": num_qubits,
        "num_blocks": num_blocks,
        "encoder_parameters": np.zeros((num_blocks, 3)),
        "decoder_parameters": np.zeros((num_blocks, 3)),
    }
    invalid_hamiltonian_parameters = {
        "rabi_frequency": 50e3,
        "omega_m": 2.75e6,
        "mu": 10000,
        "hamiltonian_type": "invalid_type",
    }

    circuit = QiskitQuantumSensingCircuit(0, circuit_parameters, invalid_hamiltonian_parameters)
    with pytest.raises(ValueError, match="Hamiltonian type 'invalid_type' is not supported."):
        circuit.run_circuit()