import optax
import pennylane.numpy as pnp
import mlflow
import os
import matplotlib.pyplot as plt

from catalyst import grad, qjit
from quantum_sensing.circuit_builder import QuantumSensingCircuit
from quantum_sensing.cost_evaluator import BayesianCostEvaluator


def run_trial(
        circuit_hyperparameters: dict,
        hamiltonian_hyperparameters: dict,
        training_hyperparameters: dict = {},
        run_name: str = None):

    evaluator = __create_cost_evaluator(circuit_hyperparameters, hamiltonian_hyperparameters)
    initial_parameters = __create_initial_parameters(circuit_hyperparameters)

    __initialize_mlflow()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(circuit_hyperparameters)
        mlflow.log_params(hamiltonian_hyperparameters)
        mlflow.log_params(training_hyperparameters)

        optimal_parameters = __run_optimization(
            evaluator,
            initial_parameters,
            training_hyperparameters,
        )

        mlflow.log_dict(optimal_parameters, "optimal_parameters.json")
        __log_mse_with_prior(evaluator, optimal_parameters)
        __log_expectation(evaluator, optimal_parameters)


def __create_cost_evaluator(circuit_hyperparameters: dict, hamiltonian_hyperparameters: dict):
    quantum_sensing_circuit = QuantumSensingCircuit(circuit_hyperparameters, hamiltonian_hyperparameters)
    evaluator = BayesianCostEvaluator(quantum_sensing_circuit, circuit_hyperparameters['num_qubits'])
    return evaluator

def __create_initial_parameters(circuit_hyperparameters: dict):
    init_circuit_parameters = pnp.random.uniform(
        low=-pnp.pi,
        high=pnp.pi,
        size=(circuit_hyperparameters['num_encoder_blocks'] + circuit_hyperparameters['num_decoder_blocks'], 3)
    )
    init_a = pnp.random.uniform(low=-pnp.pi, high=pnp.pi)

    return {
        'circuit_parameters': init_circuit_parameters,
        "a": init_a
    }

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
    mlflow.enable_system_metrics_logging()


def __run_optimization(
        evaluator: BayesianCostEvaluator,
        params: dict,
        training_hyperparameters: dict):

    learning_rate = training_hyperparameters.get('learning_rate', 0.01)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @qjit
    def qjit_compute_cost_grad(params):
        return grad(evaluator.compute_cost, method='fd')(params)
    
    num_steps = training_hyperparameters.get('num_steps', 100)
    for i in range(num_steps):
        grads = qjit_compute_cost_grad(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)

        new_params = optax.apply_updates(params, updates)
        # Update parameters
        params = new_params
        opt_state = new_opt_state

        # Log cost
        new_cost = evaluator.compute_cost(new_params)
        mlflow.log_metric("cost", float(new_cost), step=i)

    return params

def __log_mse_with_prior(cost_evaluator: BayesianCostEvaluator, optimal_parameters: dict):
    phis = cost_evaluator.get_phi_grid()
    prior_vals = cost_evaluator.get_prior()
    mse_vals = cost_evaluator.compute_mse(optimal_parameters)
    mse_db = 10 * pnp.log10(mse_vals)

    # Plotting with twin axes
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel(r"$\phi$")
    ax1.set_ylabel("MSE [dB]", color=color1)
    ax1.plot(phis, mse_db, color=color1, label="MSE (dB)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # Twin axis for prior
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Prior", color=color2)
    ax2.plot(phis, prior_vals, color=color2, linestyle='--', label="Prior")
    ax2.tick_params(axis='y', labelcolor=color2)

    # Legends
    fig.tight_layout()
    plt.title("MSE vs $\phi$ with Prior Overlay")
    mlflow.log_figure(fig, "mse_vs_phi_with_prior.png")
    plt.close(fig)

def __log_expectation(evaluator, optimal_parameters):
    circuit_parameters = optimal_parameters['circuit_parameters']
    expectations = evaluator.compute_expectation(circuit_parameters)
    phis = evaluator.get_phi_grid()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phis, expectations, label=r'$\langle J_z \rangle$', color='tab:blue', marker='o', markersize=4)

    ax.set_title(r'Expectation Value $\langle J_z \rangle$ vs. $\phi$')
    ax.set_xlabel(r'Phase $\phi$ (radians)')
    ax.set_ylabel(r'Expectation $\langle J_z \rangle$')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    mlflow.log_figure(fig, "jz_vs_phi.png")
    plt.close(fig)