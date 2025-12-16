import pytest

import numpy as np
from quantum_sensing import QuspinQuantumSensingCircuit
from quantum_sensing.cirq import CirqQuantumSensingCircuit
from quantum_sensing.pennylane import PennylaneQuantumSensingCircuit
from quantum_sensing.qiskit import QiskitQuantumSensingCircuit

hamiltonian_parameters = {
    "rabi_frequency": 50e3,
    "omega_m": 2.75e6,
    "mu": 10000,
}

@pytest.mark.parametrize("circuit_class", [
    QuspinQuantumSensingCircuit,
    PennylaneQuantumSensingCircuit,
    QiskitQuantumSensingCircuit,
    CirqQuantumSensingCircuit,
])
def test_sanity_check(circuit_class):
    '''
    All the parameterized rotations are set to zero in this test.
    This applies RY(pi/2) and RX(pi/2) on all qubits.
    When qubit starts with |0>, they end up at |+>, so all probabilities must be equally distributed.
    '''
    num_blocks = 2 
    num_qubits = 4
    circuit_parameters = {
        "num_qubits": num_qubits,
        "num_blocks": num_blocks,
        "encoder_parameters": np.zeros((num_blocks, 3)),
        "decoder_parameters": np.zeros((num_blocks, 3)),
    }

    circuit = circuit_class(0, circuit_parameters, hamiltonian_parameters)
    probabilities = circuit.run_circuit()

    np.testing.assert_allclose(probabilities, 1/(2**num_qubits), rtol=1e-5, atol=1e-8)
