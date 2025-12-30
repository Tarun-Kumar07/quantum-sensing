import multiprocessing
import os
import tempfile

import mlflow
import numpy as np
from scipy.optimize import OptimizeResult

from quantum_sensing.optimization.cost import CostEvaluator, CircuitHyperParameters, HamiltonianHyperParameters
from quantum_sensing.optimization import visualization

import jax
import jax.numpy as jnp
import optax


def __optimize(cost_evaluator, num_blocks, num_steps=300):
    # 1. Initialize parameters as a JAX dictionary (no ParamManager needed)
    params = {
        "encoder": jnp.array(np.random.uniform(-jnp.pi, jnp.pi, (num_blocks, 3))),
        "decoder": jnp.array(np.random.uniform(-jnp.pi, jnp.pi, (num_blocks, 3))),
        "a": jnp.array(1.0)
    }

    # 2. Define the Optimizer (Optax Adam)
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)

    # 3. Define the JIT-compiled update step
    @jax.jit
    def update_step(i, args):
        params, opt_state, cost_history = args

        # Compute cost and gradient in one pass
        loss_val, grads = jax.value_and_grad(lambda p: cost_evaluator.evaluate(p["encoder"], p["decoder"], p["a"]))(
            params)

        # Apply updates
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Log cost using JAX debug tools to avoid breaking JIT
        # We use jax.lax.cond to print every 10 steps
        def print_fn():
            jax.debug.print("Step: {i} | Loss: {loss_val}", i=i, loss_val=loss_val)

        jax.lax.cond(jnp.mod(i, 10) == 0, print_fn, lambda: None)

        # Update cost_history (In JAX, we update the array at index i)
        cost_history = cost_history.at[i].set(loss_val)

        return params, opt_state, cost_history

    # 4. Compile the entire optimization loop
    # We initialize a static JNP array for history
    initial_history = jnp.zeros(num_steps)
    init_args = (params, opt_state, initial_history)

    print(f"Compiling and running optimization for {num_steps} steps...")
    final_params, final_opt_state, final_history = jax.lax.fori_loop(0, num_steps, update_step, init_args)

    return final_params, float(final_history[-1]), final_history.tolist()


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

    __initialize_mlflow()

    cost_evaluator = CostEvaluator(circuit_hyperparameters, hamiltonian_hyperparameters)
    with mlflow.start_run():
        log_trial_parameters(circuit_hyperparameters, hamiltonian_hyperparameters)

        final_params, _, cost_history = __optimize(cost_evaluator, circuit_hyperparameters['num_blocks'])
        # log_result(result)

        # Visualization
        visualization.plot_cost_history(cost_history)
        phis, mse_values = cost_evaluator.evaluate_mse_for_all_phi(*final_params)
        visualization.plot_mse_with_prior(phis, mse_values)


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
