import numpy as np


def J_zig_zag(i: int, j: int, hamiltonian_parameters: dict) -> float:
    rabi_frequency = hamiltonian_parameters["rabi_frequency"]
    omega_m = hamiltonian_parameters["omega_m"]
    mu = hamiltonian_parameters["mu"]

    eta_i = 0.1 * (-1) ** i
    eta_j = 0.1 * (-1) ** j
    return (rabi_frequency ** 2) * eta_i * eta_j * mu / (mu ** 2 - omega_m ** 2)


def J_com(i: int, j: int, hyperparameters: dict) -> float:
    rabi_frequency = hyperparameters["rabi_frequency"]
    omega_m = hyperparameters["omega_m"]
    mu = hyperparameters["mu"]

    eta_i = eta_j = 0.1
    return (rabi_frequency ** 2) * eta_i * eta_j * mu / (mu ** 2 - omega_m ** 2)

def __random_perturbation_factor(maximum_value: int) -> float:
    return 0.01 * np.random.choice([-maximum_value, maximum_value])

def J_com_5(i: int, j:int, hyperparameters: dict) -> float:
    return (1 + __random_perturbation_factor(5)) * J_com(i, j, hyperparameters)

def J_com_10(i: int, j:int, hyperparameters: dict) -> float:
    return (1 + __random_perturbation_factor(10)) * J_com(i, j, hyperparameters)

def J_com_20(i: int, j:int, hyperparameters: dict) -> float:
    return (1 + __random_perturbation_factor(20)) * J_com(i, j, hyperparameters)

def get_J_function(hamiltonian_type: str):
    if hamiltonian_type == "zig_zag":
        return J_zig_zag
    elif hamiltonian_type == "com":
        return J_com
    elif hamiltonian_type == "com_5":
        return J_com_5
    elif hamiltonian_type == "com_10":
        return J_com_10
    elif hamiltonian_type == "com_20":
        return J_com_20
    else:
        raise ValueError(f"Hamiltonian type '{hamiltonian_type}' is not supported.")


def interaction_strength(i: int, j: int, hamiltonian_parameters: dict) -> float:
    j_function = get_J_function(hamiltonian_parameters["hamiltonian_type"])
    return j_function(i, j, hamiltonian_parameters)
