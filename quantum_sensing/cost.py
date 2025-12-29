import numpy as np
from scipy.integrate import simpson


def prior_wrapped_gaussian(phi, delta=0.3, k_max=5):
    return sum(np.exp(-((phi + 2 * np.pi * k) ** 2) / (2 * delta ** 2)) for k in range(-k_max, k_max + 1)) / (
            np.sqrt(2 * np.pi) * delta)


def state_to_magnetization(state, num_qubits):
    hamming_weight = bin(state).count('1')
    return num_qubits - 2 * hamming_weight


def mean_square_error(phi, probabilities, a):
    mse = 0.0
    num_qubits = int(np.log2(len(probabilities)))
    for state, p in enumerate(probabilities):
        m = state_to_magnetization(state, num_qubits)
        phi_est = a * m
        mse += ((phi_est - phi) ** 2) * p
    return mse


def cost_function(phi_signal, probabilities_for_all_phi, a):
    cost_for_phi = []
    for phi, probabilities in zip(phi_signal, probabilities_for_all_phi):
        cost = mean_square_error(phi, probabilities, a) * prior_wrapped_gaussian(phi)
        cost_for_phi.append(cost)

    return simpson(cost_for_phi, phi_signal)
