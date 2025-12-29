import abc
import numpy as np

from quantum_sensing.circuit.hamiltonian_interaction_strength import get_J_function


class QuantumSensingCircuit(abc.ABC):
    def __init__(self, phi_signal, circuit_parameters: dict, hamiltonian_parameters: dict):
        self.__num_qubits = circuit_parameters["num_qubits"]
        self.__num_blocks = circuit_parameters["num_blocks"]
        # TODO verify shapes of encoder and decoder parameters, will be useful when saving
        self.__encoder_parameters = circuit_parameters["encoder_parameters"]
        self.__decoder_parameters = circuit_parameters["decoder_parameters"]
        self.__phi_signal = phi_signal
        self.__hamiltonian_parameters = hamiltonian_parameters

    def run_circuit(self) -> np.ndarray:
        """
        :return: probability dictionary of binary representation of states to their probabilities
        """
        qubit_pairs = [(i, j) for i in range(self.__num_qubits) for j in range(i + 1, self.__num_qubits)]
        j_function = get_J_function(self.__hamiltonian_parameters.get("hamiltonian_type", "zig_zag"))
        interaction_strengths = [(j_function(i, j, self.__hamiltonian_parameters), i, j) for i, j in qubit_pairs]

        self.single_body_interaction(np.pi / 2, 'y', self.__num_qubits)

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

        self.single_body_interaction(np.pi / 2, 'x', self.__num_qubits)

        return self.calculate_probabilities()

    @abc.abstractmethod
    def single_body_interaction(self, theta: float, operator: str, num_qubits: int):
        """
        Implements the unitary evolution operator defined by:
            exp(-i * θ * Σ_{i} (operator_i))
        where operator_i is operator acting on ith qubit    
        :param theta: Angle of rotation
        :param operator: Can be 'x', 'y' or 'z'
        :param num_qubits: Number of qubits in the circuit, so that the operation can be applied to all qubits
        :return: None
        """
        pass

    @abc.abstractmethod
    def double_body_interaction(self, theta: float, operator: str, interaction_strengths: list[tuple]):
        """
        Implements the unitary evolution operator defined by:
            exp(-i * θ * Σ_{i,j} [J_ij * (operator_i ⊗ operator_j)])
        where operator_i is operator acting on ith qubit, J_ij can be derived from interaction_strengths

        :param theta: Angle of rotation
        :param operator: Can be 'x', 'y' or 'z'
        :param interaction_strengths: List of tuples (J_ij, i, j) where i and j are qubit indices and J_ij is
               the interaction strength
        :return: None
        """
        pass

    @abc.abstractmethod
    def calculate_probabilities(self) -> np.ndarray:
        """
        :return: probability array of the final state after running the circuit
        """
        pass


def create_quantum_sensing_circuit(
        phi_signal: float,
        circuit_parameters: dict,
        hamiltonian_parameters: dict,
        backend: str = 'pennylane') -> QuantumSensingCircuit:

    if backend == "qiskit":
        from quantum_sensing.circuit.qiskit import QiskitQuantumSensingCircuit
        return QiskitQuantumSensingCircuit(phi_signal, circuit_parameters, hamiltonian_parameters)
    elif backend == "cirq":
        from quantum_sensing.circuit.cirq import CirqQuantumSensingCircuit
        return CirqQuantumSensingCircuit(phi_signal, circuit_parameters, hamiltonian_parameters)
    elif backend == "quspin":
        from quantum_sensing.circuit.quspin import QuspinQuantumSensingCircuit
        return QuspinQuantumSensingCircuit(phi_signal, circuit_parameters, hamiltonian_parameters)
    elif backend == "pennylane":
        from quantum_sensing.circuit.pennylane import PennyLaneQuantumSensingCircuit
        return PennyLaneQuantumSensingCircuit(phi_signal, circuit_parameters, hamiltonian_parameters)
    else:
        raise ValueError(f"Backend '{backend}' is not supported.")
