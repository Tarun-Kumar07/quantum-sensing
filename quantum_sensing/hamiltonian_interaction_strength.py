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


def get_J_function(hamiltonian_type: str):
    if hamiltonian_type == "zig_zag":
        return J_zig_zag
    elif hamiltonian_type == "com":
        return J_com
    else:
        raise ValueError(f"Hamiltonian type '{hamiltonian_type}' is not supported.")


def interaction_strength(i: int, j: int, hamiltonian_parameters: dict) -> float:
    j_function = get_J_function(hamiltonian_parameters["hamiltonian_type"])
    return j_function(i, j, hamiltonian_parameters)
