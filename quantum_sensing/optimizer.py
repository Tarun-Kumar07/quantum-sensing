from typing import TypedDict

import os

import mlflow as mlflow
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from quantum_sensing.circuit import create_quantum_sensing_circuit
from quantum_sensing.cost import cost_function, mean_square_error, prior_wrapped_gaussian

PHI_RANGE = np.linspace(-np.pi, np.pi, 100)


class CircuitHyperParameters(TypedDict):
    num_qubits: int
    num_blocks: int


class HamiltonianHyperParameters(TypedDict):
    hamiltonian_type: str
    rabi_frequency: float
    omega_m: float
    mu: float


def unflatten_parameters(x, num_blocks):
    i = 0
    number_of_parameters_per_block = 3
    encoder_parameters = x[i: i + number_of_parameters_per_block * num_blocks].reshape(num_blocks,
                                                                                       number_of_parameters_per_block)
    i += number_of_parameters_per_block * num_blocks
    decoder_parameters = x[i: i + number_of_parameters_per_block * num_blocks].reshape(num_blocks,
                                                                                       number_of_parameters_per_block)
    i += number_of_parameters_per_block * num_blocks
    a = x[i]
    return encoder_parameters, decoder_parameters, a


def create_initial_guess(circuit_hyperparameters):
    num_blocks = circuit_hyperparameters['num_blocks']
    block_params = num_blocks * 3

    # Encoder + Decoder + a
    number_of_params = 2 * block_params + 1

    return np.random.uniform(-np.pi, np.pi, number_of_params)


def create_cost_function(circuit_hyperparameters, hamiltonian_hyperparameters):
    def cost_function_wrapped(x):
        encoder_parameters, decoder_parameters, a = unflatten_parameters(x, circuit_hyperparameters['num_blocks'])
        circuit_parameters = {
            "num_qubits": circuit_hyperparameters['num_qubits'],
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": decoder_parameters,
        }

        probabilities = []
        for phi in PHI_RANGE:
            circuit = create_quantum_sensing_circuit(phi, circuit_parameters, hamiltonian_hyperparameters)
            probabilities.append(circuit.run_circuit())

        return cost_function(PHI_RANGE, probabilities, a)

    return cost_function_wrapped


def __initialize_mlflow():
    tracking_directory = os.path.abspath("../data/mlflow")
    os.makedirs(tracking_directory, exist_ok=True)

    db_path = os.path.join(tracking_directory, "mlflow.db")
    artifact_path = os.path.join(tracking_directory, "artifacts")

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    experiment_name = "Quantum_Sensing"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=f"file://{artifact_path}"
        )

    mlflow.set_experiment(experiment_name)


def run_trial(
        circuit_hyperparameters: CircuitHyperParameters,
        hamiltonian_hyperparameters: HamiltonianHyperParameters):
    __initialize_mlflow()
    with mlflow.start_run():
        __log_hyperparameters(circuit_hyperparameters, hamiltonian_hyperparameters)

        cost_function_to_optimize = create_cost_function(circuit_hyperparameters, hamiltonian_hyperparameters)
        initial_guess = create_initial_guess(circuit_hyperparameters)

        cost_history = []

        def cost_history_logger(flattened):
            cost = cost_function_to_optimize(flattened)
            cost_history.append(cost)
            return cost

        options = {
            'maxiter': 300,
            'disp': True,
            'fatol': 1e-3
        }

        optimized_result = minimize(cost_history_logger, initial_guess, method='Nelder-Mead', options=options)

        mlflow.log_param("optimization_success", optimized_result.success, True)
        mlflow.log_metric("final_cost", optimized_result.fun, True)
        save_cost_history(cost_history)
        optimal_encoder_parameters, optimal_decoder_parameters, optimized_a = unflatten_parameters(optimized_result.x, circuit_hyperparameters['num_blocks'])
        optimal_circuit_parameters = {
            "num_qubits": circuit_hyperparameters['num_qubits'],
            "encoder_parameters": optimal_encoder_parameters,
            "decoder_parameters": optimal_decoder_parameters,
        }
        save_mse_plot(optimal_circuit_parameters, optimized_a, hamiltonian_hyperparameters)
        mlflow.log_dict(optimal_circuit_parameters, "optimal_circuit_parameters.json")


def save_cost_history(cost_history):
    figure = plt.figure(figsize=(6, 4))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost function value')
    plt.title('Optimization Cost History')
    plt.grid(True)
    plt.tight_layout()
    mlflow.log_figure(figure, "optimization_cost_history.png")
    plt.close(figure)


def save_mse_plot(optimal_circuit_parameters, optimized_a, hamiltonian_hyperparameters):
    mse_values = []
    for phi in PHI_RANGE:
        optimal_circuit = create_quantum_sensing_circuit(phi, optimal_circuit_parameters, hamiltonian_hyperparameters)
        probabilities = optimal_circuit.run_circuit()
        mse = mean_square_error(phi, probabilities, optimized_a)
        mse_values.append(mse)

    mse_db = 10 * np.log10(mse_values)
    prior_vals = np.array([prior_wrapped_gaussian(phi) for phi in PHI_RANGE])

    # Plotting with twin axes
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel(r"$\phi$")
    ax1.set_ylabel("MSE [dB]", color=color1)
    ax1.plot(PHI_RANGE, mse_db, color=color1, label="MSE (dB)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # Twin axis for prior
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Prior", color=color2)
    ax2.plot(PHI_RANGE, prior_vals, color=color2, linestyle='--', label="Prior")
    ax2.tick_params(axis='y', labelcolor=color2)

    # Legends
    fig.tight_layout()
    plt.title("MSE vs $\phi$ with Prior Overlay")
    mlflow.log_figure(fig, "mse_vs_phi_with_prior.png")
    plt.close(fig)


def __log_hyperparameters(circuit_hyperparameters, hamiltonian_hyperparameters):
    for key, value in circuit_hyperparameters.items():
        mlflow.log_param(key, value, True)
    for key, value in hamiltonian_hyperparameters.items():
        mlflow.log_param(key, value, True)
