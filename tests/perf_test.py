import pytest
import tracemalloc
import time
import numpy as np
from quantum_sensing import QuspinQuantumSensingCircuit
from quantum_sensing.cirq import CirqQuantumSensingCircuit
from quantum_sensing.pennylane import PennylaneQuantumSensingCircuit
from quantum_sensing.qiskit import QiskitQuantumSensingCircuit

# -------------------------------
# Parameter values
# -------------------------------
hamiltonian_parameters = {
    "rabi_frequency": 50e3,
    "omega_m": 2.75e6,
    "mu": 10000,
}

def generate_circuit_parameters(num_qubits, num_blocks):
    return {
        "num_qubits": num_qubits,
        "num_blocks": num_blocks,
        "encoder_parameters": np.full((num_blocks, 3), np.pi/4),
        "decoder_parameters": np.full((num_blocks, 3), np.pi/4),
    }

# -------------------------------
# Benchmark test using pytest API
# -------------------------------
@pytest.mark.parametrize("num_qubits", list(range(4, 5)))
@pytest.mark.parametrize("num_blocks", list(range(1, 2)))
@pytest.mark.parametrize("circuit_class", [
    QuspinQuantumSensingCircuit,
    PennylaneQuantumSensingCircuit,
    QiskitQuantumSensingCircuit,
    CirqQuantumSensingCircuit,
])
def test_benchmark_circuit(
        circuit_class,
        num_qubits,
        num_blocks,
        record_property):
    """
    Benchmark multiplication functions for time + memory.
    Stores results using pytest's record_property API.
    """
    circuit_parameters = generate_circuit_parameters(num_qubits, num_blocks)

    # Measure memory + time
    tracemalloc.start()
    start_time = time.perf_counter()

    phi_signal = np.pi/4
    circuit = circuit_class(phi_signal, circuit_parameters, hamiltonian_parameters)
    probabilties = circuit.run_circuit()

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # print(probabilties)
    assert probabilties.shape == (2 ** num_qubits,)

    # Record metrics in pytest report
    record_property("circuit_class", circuit_class.__name__)
    record_property("num_qubits", num_qubits)
    record_property("num_blocks", num_blocks)
    record_property("time_sec", end_time - start_time)
    record_property("peak_memory_bytes", peak)
