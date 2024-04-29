from paddle import Tensor, to_tensor
from paddle_quantum.mbqc.qobject import State
from paddle_quantum.mbqc.utils import zero_state, one_state, plus_state, minus_state
from paddle_quantum.mbqc.utils import h_gate, pauli_gate, rotation_gate
from paddle_quantum.mbqc.utils import cnot_gate, cz_gate
from paddle_quantum.mbqc.utils import kron, basis
from paddle_quantum.mbqc.utils import to_projector
from paddle import t, matmul, conj, real, reshape, multiply, trace

from paddle_quantum.qinfo import state_fidelity, partial_trace

from numpy import log2

from qiskit.quantum_info import random_density_matrix, random_statevector

def random_rho_matrix(n):
    return to_tensor(random_density_matrix(2 ** n).data.reshape(2 ** n, 2 ** n),  dtype='complex128')

def depolaring_channel(rho, beta):
    dim = int(log2(rho.shape[0]))
    identity = kron([pauli_gate('I') for _ in range(dim)])
    return (1 - beta) * rho + beta / dim * identity
def depolarized_Bell_state(beta):
    zero_density = to_projector(zero_state())
    # Hadamard gate
    H_zero_density = matmul(matmul(h_gate(), zero_density), h_gate())
    # CNOT gate
    Bell_state = matmul(matmul(cnot_gate(), kron([H_zero_density, zero_density])), cnot_gate())
    # Add depolarizing channel
    return depolaring_channel(Bell_state, beta)

def basis_to_observable(basis_int):
    if basis_int == 0:
        return (pauli_gate('I') + pauli_gate('Z')) / 2
    elif basis_int == 1:
        return (pauli_gate('I') - pauli_gate('Z')) / 2
    else:
        raise ValueError()
def tensor_projective_measurement(rho):
    eps = 10 ** (-10)
    prob = [0, 0, 0, 0]
    state_after_measurement = [0, 0, 0, 0]

    results = ['00', '01,', '10', '11']
    for i, result in enumerate(results):
        basis_q0 = int(result[0])
        basis_q1 = int(result[1])
        obs0 = basis_to_observable(basis_q0)
        obs1 = basis_to_observable(basis_q1)
        observable = kron([obs0, obs1, pauli_gate('I')])
        prob_result = float(trace(matmul(rho, observable)).numpy())
        prob[i] = prob_result
        state_after = partial_trace(matmul(rho, observable), 2 ** 2, 2, 1)
        # Byproduct correction
        if basis_q0 == 1:
            state_after = matmul(matmul(kron([pauli_gate('I'), pauli_gate('I'), pauli_gate('Z')]), state_after),   kron([pauli_gate('I'), pauli_gate('I'), pauli_gate('Z')]))
        if basis_q1 == 1:
            state_after = matmul(matmul(kron([pauli_gate('I'), pauli_gate('I'), pauli_gate('X')]), state_after),   kron([pauli_gate('I'), pauli_gate('I'), pauli_gate('X')]))
        state_after_measurement[i] = state_after
    return state_after_measurement

























    # # Calculate the probability and post-measurement states
    # for result in [0, 1]:
    #     basis_dagger = t(conj(basis_list[result]))
    #     # Reshape the state, multiply the basis and reshape it back
    #     state_unnorm[result] = reshape(matmul(basis_dagger,
    #                                           reshape(new_bg_state.vector, [2, half_length])), [half_length, 1])
    #     probability = matmul(t(conj(state_unnorm[result])), state_unnorm[result])
    #     is_complex128 = probability.dtype == to_tensor([], dtype='complex128').dtype
    #     prob[result] = real(probability) if is_complex128 else probability
    #
    # # Randomly choose a result and its corresponding post-measurement state
    #     if prob[0].numpy().item() < eps:
    #         result = 1
    #         post_state_vector = state_unnorm[1]
    #     elif prob[1].numpy().item() < eps:
    #         result = 0
    #         post_state_vector = state_unnorm[0]
    #     else:  # Take a random choice of outcome
    #         result = random.choice(2, 1, p=[prob[0].numpy().item(), prob[1].numpy().item()]).item()
    #         # Normalize the post-measurement state
    #         post_state_vector = state_unnorm[result] / prob[result].sqrt()
#
# rho = kron([to_projector(zero_state()), to_projector(zero_state()), to_projector(one_state())])
# eps = 10 ** (-10)
# prob = [0, 0, 0, 0]
# state_after_measurement = [0, 0, 0, 0]
#
# results = ['00', '01,', '10', '11']
# for i, result in enumerate(results):
#     basis_q0 = int(result[0])
#     basis_q1 = int(result[1])
#     obs0 = basis_to_observable(basis_q0)
#     obs1 = basis_to_observable(basis_q1)
#     observable = kron([obs0, obs1, pauli_gate('I')])
#     prob[i] = float(trace(matmul(rho, observable)).numpy())
#     state_after = partial_trace(matmul(rho, observable), 2 ** 2, 2, 1)
#     # Byproduc correction
#     if basis_q0 == 1:
#         state_after = matmul(matmul(kron([pauli_gate('I'), pauli_gate('I'), pauli_gate('Z')]), state_after),
#                              kron([pauli_gate('I'), pauli_gate('I'), pauli_gate('Z')]))
#     if basis_q1 == 1:
#         state_after = matmul(matmul(kron([pauli_gate('I'), pauli_gate('I'), pauli_gate('X')]), state_after),
#                              kron([pauli_gate('I'), pauli_gate('I'), pauli_gate('X')]))
#     state_after_measurement[i] = state_after
#
# print(state_after_measurement)



















# def teleportation(rho, beta):
#     # Depolarized Bell state
#     Noisy_Bell_state = depolarized_Bell_state(beta)
#     input_state = kron([rho, Noisy_Bell_state])
#     # Entangling
#     entangled_state = matmul(matmul(kron([cnot_gate(), pauli_gate('I')]), input_state), kron([cnot_gate(), pauli_gate('I')]))
#     # Hadamard gate
#     state_before_measurement = matmul(matmul(kron([h_gate(), pauli_gate('I'), pauli_gate('I')]), entangled_state), kron([h_gate(), pauli_gate('I'), pauli_gate('I')]))
#
#
#
#
#
#     zero_density = to_projector(zero_state())
#     # Hadamard gate
#     H_zero_density = matmul(matmul(h_gate(),  zero_density), h_gate())
#
#
#
#     return teleport_rho



n = 5
beta = 0.1
rho = random_rho_matrix(n)
Bell_state = depolarized_Bell_state(beta)





