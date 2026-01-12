from quantum_sensing.hamiltonian_interaction_strength import interaction_strength


def test_J_com_5_returns_same_value_for_same_inputs():
    hamiltonian_hyperparameters = {
        "rabi_frequency": 1.0,
        "omega_m": 2.0,
        "mu": 3.0,
        "hamiltonian_type": "com_5"
    }

    values = [interaction_strength(3, 1, hamiltonian_hyperparameters) for _ in range(100)]

    assert all(value == values[0] for value in values), "J_com_5 returned different values for the same inputs"