import pytest
import tracemalloc
import time
import os
import numpy as np
from quantum_sensing.circuit import create_quantum_sensing_circuit_random

# -------------------------------
# Parameter values
# -------------------------------
hamiltonian_parameters = {
    "rabi_frequency": 50e3,
    "omega_m": 2.75e6,
    "mu": 10000,
}

# -------------------------------
# Benchmark test using pytest API
# -------------------------------
benchmark_params = list(set(
    [(q, 5) for q in range(5, 14)] +
    [(13, b) for b in range(1, 6)]
))
benchmark_params.sort()


@pytest.mark.parametrize("num_qubits, num_blocks", benchmark_params)
@pytest.mark.parametrize("backend", ['quspin', 'qiskit', 'pennylane', 'cirq'])
def test_benchmark_circuit(
        num_qubits,
        num_blocks,
        backend,
        record_property):
    tracemalloc.start()
    start_time = time.perf_counter()

    phi_signal = np.pi / 4
    circuit = create_quantum_sensing_circuit_random(num_qubits, num_blocks, hamiltonian_parameters, phi_signal, backend)
    probabilties = circuit.run_circuit()

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # print(probabilties)
    assert probabilties.shape == (2 ** num_qubits,)

    record_property("circuit_class", circuit.__class__.__name__)
    record_property("num_qubits", num_qubits)
    record_property("num_blocks", num_blocks)
    record_property("num_threads", os.environ.get("OMP_NUM_THREADS"))
    record_property("time_sec", end_time - start_time)
    record_property("peak_memory_bytes", peak)
