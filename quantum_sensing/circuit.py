import abc
import numpy as np

from quantum_sensing.hamiltonian_interaction_strength import J_zig_zag


class QuantumSensingCircuit(abc.ABC):
    def __init__(self, phi_signal, circuit_parameters: dict, hamiltonian_parameters: dict):
        self.__num_qubits = circuit_parameters["num_qubits"]
        self.__num_blocks = circuit_parameters["num_blocks"]
        # TODO verify shapes of encoder and decoder parameters, will be useful when saving
        self.__encoder_parameters = circuit_parameters["encoder_parameters"]
        self.__decoder_parameters = circuit_parameters["decoder_parameters"]
        self.__phi_signal = phi_signal

        self.__hamiltonian_parameters = hamiltonian_parameters

    def run_circuit(self) -> dict:
        """
        :return: probability dictionary of binary representation of states to their probabilities
        """
        # TODO parameterize this, right now hardcoded to zig zag
        qubit_pairs = [(i, j) for i in range(self.__num_qubits) for j in range(i + 1, self.__num_qubits)]
        interaction_strengths = [(J_zig_zag(i, j, self.__hamiltonian_parameters), i, j) for i, j in qubit_pairs]

        self.single_body_interaction(np.pi/2, 'y', self.__num_qubits)

        # Encoder block
        for block in range(self.__num_blocks):
            single_rotation, xx_rotation, zz_rotation = self.__encoder_parameters[block]
            self.single_body_interaction(single_rotation, 'x', self.__num_qubits)
            self.double_body_interaction(xx_rotation, 'x', interaction_strengths)
            self.double_body_interaction(zz_rotation, 'z', interaction_strengths)

        # Sensing layer
        self.single_body_interaction(self.__phi_signal, 'z', self.__num_qubits)

        # Decoder block
        for block in range(self.__num_blocks):
            single_rotation, xx_rotation, zz_rotation = self.__decoder_parameters[block]
            self.single_body_interaction(single_rotation, 'x', self.__num_qubits)
            self.double_body_interaction(xx_rotation, 'x', interaction_strengths)
            self.double_body_interaction(zz_rotation, 'z', interaction_strengths)

        return self.calculate_probabilities()

    @abc.abstractmethod
    def single_body_interaction(self, theta: float, operator: str, num_qubits: int):
        """
        :param theta: Angle of rotation
        :param operator: Can be 'x', 'y' or 'z'
        :param num_qubits: Number of qubits in the circuit, so that the operation can be applied to all qubits
        :return: None
        """
        pass

    @abc.abstractmethod
    def double_body_interaction(self, theta: float, operator: str, interaction_strengths: list[tuple]):
        """
        :param theta: Angle of rotation
        :param operator: Can be 'x', 'y' or 'z'
        :param interaction_strengths: List of tuples (J_ij, i, j) where i and j are qubit indices and J_ij is
               the interaction strength
        :return: None
        """
        pass

    @abc.abstractmethod
    def calculate_probabilities(self) -> dict:
        """
        :return: probability array of the final state after running the circuit
        """
        pass
