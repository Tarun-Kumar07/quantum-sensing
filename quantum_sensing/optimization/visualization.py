import matplotlib.pyplot as plt
import numpy as np
import mlflow

from quantum_sensing.optimization.cost import prior_wrapped_gaussian


def plot_cost_history(cost_history):
    figure = plt.figure(figsize=(6, 4))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost function value')
    plt.title('Optimization Cost History')
    plt.grid(True)
    plt.tight_layout()
    mlflow.log_figure(figure, "optimization_cost_history.png")
    plt.close(figure)


def plot_mse_with_prior(phis, mse_values):
    prior_vals = np.array([prior_wrapped_gaussian(phi) for phi in phis])
    mse_db = 10 * np.log10(mse_values)

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
