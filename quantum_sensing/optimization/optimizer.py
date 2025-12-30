import os
import tempfile

import mlflow
import numpy as np
import psutil
from numpy import ndarray
from scipy.optimize import minimize, OptimizeResult

from quantum_sensing.optimization.cost import CostEvaluator, CircuitHyperParameters, HamiltonianHyperParameters
from quantum_sensing.optimization import visualization


class ParameterManager:
    """Handles flattening and unflattening of optimization parameters."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.params_per_block = 3
        # Total params = (Encoder params) + (Decoder params) + (scalar a)
        self.total_params = (self.params_per_block * self.num_blocks * 2) + 1

    def create_initial_guess(self) -> np.ndarray:
        return np.random.uniform(-np.pi, np.pi, self.total_params)

    def unflatten(self, x: np.ndarray) -> tuple[ndarray, ndarray, float]:
        block_size = self.params_per_block * self.num_blocks

        encoder_flat = x[0:block_size]
        decoder_flat = x[block_size:2 * block_size]
        a = x[-1]

        encoder_params = encoder_flat.reshape(self.num_blocks, self.params_per_block)
        decoder_params = decoder_flat.reshape(self.num_blocks, self.params_per_block)

        return encoder_params, decoder_params, a


def __optimize(cost_evaluator: CostEvaluator, param_manager: ParameterManager) -> tuple[OptimizeResult, list[float]]:
    """Runs the pure optimization logic."""
    initial_guess = param_manager.create_initial_guess()
    cost_history = []

    def objective_wrapper(x):
        encoder_parameters, decoder_parameters, a = param_manager.unflatten(x)
        cost = cost_evaluator.evaluate(encoder_parameters, decoder_parameters, a)
        cost_history.append(cost)
        return cost

    result = minimize(
        objective_wrapper,
        initial_guess,
        method='Nelder-Mead',
        options={
            'maxiter': 300,
            'disp': True,
            'fatol': 1e-3
        }
    )

    return result, cost_history


# --- MLflow Integration ---

def __initialize_mlflow():
    tracking_directory = os.path.abspath("../data/mlflow")
    os.makedirs(tracking_directory, exist_ok=True)
    db_path = os.path.join(tracking_directory, "mlflow.db")
    artifact_path = os.path.join(tracking_directory, "artifacts")

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    experiment_name = "Quantum_Sensing"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name, artifact_location=f"file://{artifact_path}")
    mlflow.set_experiment(experiment_name)


def run_trial(circuit_hyperparameters: CircuitHyperParameters, hamiltonian_hyperparameters: HamiltonianHyperParameters):
    """Orchestrates the experiment with MLflow logging."""
    cores_to_use = get_available_cpu_cores()
    cost_evaluator = CostEvaluator(circuit_hyperparameters, hamiltonian_hyperparameters, cores_to_use)
    param_manager = ParameterManager(circuit_hyperparameters['num_blocks'])

    __initialize_mlflow()
    with mlflow.start_run():
        log_trial_parameters(circuit_hyperparameters, hamiltonian_hyperparameters)

        result, cost_history = __optimize(cost_evaluator, param_manager)
        log_result(result)
        optimal_encoder_parameters, optimal_decoder_parameters, optimal_a = param_manager.unflatten(result.x)

        # Visualization
        visualization.plot_cost_history(cost_history)
        phis, mse_values = cost_evaluator.evaluate_mse_for_all_phi(
            optimal_encoder_parameters,
            optimal_decoder_parameters,
            optimal_a)
        visualization.plot_mse_with_prior(phis, mse_values)


def get_available_cpu_cores():
    physical_cpu_cores = psutil.cpu_count(logical=False) or os.cpu_count()
    # Subtracting 2 to leave some cores free for other processes
    cores_to_use = max(1, physical_cpu_cores - 2)
    print(f"Using {cores_to_use} out of {physical_cpu_cores} physical CPU cores.")
    return cores_to_use


def log_trial_parameters(circuit_hyperparameters, hamiltonian_hyperparameters):
    for k, v in circuit_hyperparameters.items():
        mlflow.log_param(k, v)
    for k, v in hamiltonian_hyperparameters.items():
        mlflow.log_param(k, v)


def log_result(result: OptimizeResult):
    mlflow.log_param("optimization_success", result.success)
    mlflow.log_metric("final_cost", result.fun)
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = os.path.join(tmp_dir, "parameters.npy")
        np.save(temp_path, result.x)
        mlflow.log_artifact(temp_path)
