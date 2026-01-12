from quantum_sensing.hamiltonian_interaction_strength import interaction_strength


def test_J_com_5_returns_different_value_for_same_inputs():
    hamiltonian_hyperparameters = {
        "rabi_frequency": 1.0,
        "omega_m": 2.0,
        "mu": 3.0,
        "hamiltonian_type": "com_5"
    }

    j_12 = interaction_strength(1, 2, hamiltonian_hyperparameters)
    j_34 = interaction_strength(3, 4, hamiltonian_hyperparameters)

    assert j_12 != j_34, "J_com_5 returned the same value for different qubit pairs"