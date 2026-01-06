import pennylane as qml
import pennylane.numpy as pnp
from catalyst import qjit

from quantum_sensing.hamiltonian_interaction_strength import interaction_strength


class QuantumSensingCircuit:
    def __init__(self, circuit_params:dict, hamiltonian_params:dict):
        self.__num_qubits = circuit_params['num_qubits']
        self.__num_encoder_blocks = circuit_params['num_encoder_blocks']
        self.__num_decoder_blocks = circuit_params['num_decoder_blocks']

        qubit_pairs = [(i, j) for i in range(self.__num_qubits) for j in range(i + 1, self.__num_qubits)]
        self.__interaction_strengths = [(interaction_strength(i, j, hamiltonian_params), i, j) for i, j in qubit_pairs]

        dev = qml.device("lightning.qubit", wires=self.__num_qubits)
        self.compute_probabilities = qjit(qml.QNode(self._probability_logic, dev))
        self.compute_expectation = qjit(qml.QNode(self._expectation_logic, dev))

    def get_num_qubits(self):
        return self.__num_qubits

    def _probability_logic(self, phi, circuit_parameters):
        self.__circuit_body(phi, circuit_parameters)
        return qml.probs(wires=range(self.__num_qubits))

    def _expectation_logic(self, phi, circuit_parameters):
        self.__circuit_body(phi, circuit_parameters)
        obs = [qml.PauliZ(i) for i in range(self.__num_qubits)]
        Jz_op = 0.5 * qml.sum(*obs)

        return qml.expval(Jz_op)

    def __circuit_body(self, phi: float, block_parameters) -> None:
        if block_parameters.shape != (self.__num_encoder_blocks + self.__num_decoder_blocks, 3):
            raise ValueError(f"Block parameters have incorrect shape, expected : "
                             f"({self.__num_encoder_blocks + self.__num_decoder_blocks}, 3)")

        encoder_params = block_parameters[:self.__num_encoder_blocks]
        decoder_params = block_parameters[self.__num_encoder_blocks:]
        self.__apply_single_qubit_rotations(qml.RY, pnp.pi / 2)

        for i in range(len(encoder_params)):
            zz_rot, xx_rot, s_rot = encoder_params[i]
            self.__apply_qubit_interaction(qml.IsingZZ, zz_rot)
            self.__apply_qubit_interaction(qml.IsingXX, xx_rot)
            self.__apply_single_qubit_rotations(qml.RX, s_rot)

        self.__apply_single_qubit_rotations(qml.RZ, phi)

        for i in range(len(decoder_params)):
            zz_rot, xx_rot, s_rot = decoder_params[i]
            self.__apply_single_qubit_rotations(qml.RX, s_rot)
            self.__apply_qubit_interaction(qml.IsingXX, xx_rot)
            self.__apply_qubit_interaction(qml.IsingZZ, zz_rot)

        self.__apply_single_qubit_rotations(qml.RX, pnp.pi / 2)

    def __apply_single_qubit_rotations(self, gate, theta: float) -> None:
        for i in range(self.__num_qubits):
            gate(theta, wires=i)

    def __apply_qubit_interaction(self, gate, theta: float) -> None:
        for J_ij, i, j in self.__interaction_strengths:
            gate(theta * J_ij, wires=[i, j])
