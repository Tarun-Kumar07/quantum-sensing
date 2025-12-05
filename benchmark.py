from itertools import product

import numpy as np
import timeit

from circuit.quspin import run_circuit

hamiltonian_parameters = {
    "rabi_frequency": 50e3,
    "omega_m": 2.75e6,
    "mu": 10000,
}

def generate_circuit_parameters(num_qubits, num_blocks):
    return {
        "num_qubits": num_qubits,
        "num_blocks": num_blocks,
        "encoder_parameters": np.random.uniform(-np.pi, np.pi, size = (num_blocks, 2)),
        "decoder_parameters": np.random.uniform(-np.pi, np.pi, size = (num_blocks, 2)),
        "phi": np.random.uniform(-np.pi, np.pi),
    }

def benchmark(name):
    qubits = list(range(4, 13))
    blocks = list(range(2, 5))
    filename = f"./benchmark.txt"

    with open(filename, "w") as file:
        for qubits, blocks in list(product(qubits, blocks)):
            circuit_params = generate_circuit_parameters(qubits, blocks)
            execution_time = timeit.timeit(lambda: run_circuit(circuit_params, hamiltonian_parameters), number=10)
            file.write(f"{name} - Qubits: {qubits}, Blocks: {blocks}, Average Execution time 10 runs: {execution_time/10} seconds \n")

if __name__ == '__main__':
    benchmark("quspin")