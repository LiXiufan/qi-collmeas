from numpy import array, identity
from numpy import log, exp, cos, sin, conj, transpose, sqrt
from numpy import random, pi, linspace
from numpy import trace, real
from numpy import kron, eye, diag, log2
from numpy import linalg
from qiskit.quantum_info import random_density_matrix, random_statevector

import matplotlib.pyplot as plt

eps = 1e-12

def ZeroState():
    return array([1, 0]).reshape(-1, 1)
def OneState():
    return array([0, 1]).reshape(-1, 1)
def PlusState():
    return 1 / sqrt(2) * array([1, 1]).reshape(-1, 1)
def MinusState():
    return 1 / sqrt(2) * array([1, -1]).reshape(-1, 1)

def I():
    return identity(2)
def X():
    return array([[0, 1], [1, 0]])
def Y():
    return array([[0, -1j], [1j, 0]])
def Z():
    return array([[1, 0], [0, -1]])
def H():
    return 1 / sqrt(2) * array([[1, 1], [1, -1]])
def Rx(theta):
    return cos(theta / 2) * I() - 1j * sin(theta / 2) * X()
def Ry(theta):
    return cos(theta / 2) * I() - 1j * sin(theta / 2) * Y()
def Rz(theta):
    return cos(theta / 2) * I() - 1j * sin(theta / 2) * Z()
def u3(theta,phi,lam):
    #Construct an arbitrary single qubit unitary
    return Rz(lam) @ Ry(phi) @ Rz(theta)
def CNOT():
    return array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
def Nkron(matrices):
    dim = len(matrices)
    if dim == 1:
        kronMatrix = matrices[0]
    else:
        kronMatrix = identity(1)
        for i in range(dim):
            kronMatrix = kron(kronMatrix, matrices[i])
    return kronMatrix


def permute_to_front(state, which_system):
    r"""Move a subsystem of a system to the first.

    Assume that a quantum state :math:`\psi\rangle` can be decomposed to tensor product form:

    .. math::

        |\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes |\psi_3\rangle \otimes \cdots


    the labels of each :math:`|\psi_i\rangle` is :math:`i` , so the total labels of the current system are:

    .. math::

        \text{label} = \{1, 2, 3, \cdots \}

    Assume that the label of the subsystem to be moved is: i

    The output new quantum state is:

    .. math::

        |\psi_i\rangle \otimes |\psi_1\rangle \otimes \cdots |\psi_{i-1}\rangle \otimes |\psi_{i+1}\rangle \otimes \cdots

    Args:
        state (State): the quantum state to be processed
        which_system (str): the labels of the subsystem to be moved.

    Returns:
        State: the final state after the move operation.
    """
    assert which_system in state.system, 'the system to permute must be in the state systems.'
    system_idx = state.system.index(which_system)
    if system_idx == 0:  # system in the front
        return state
    elif system_idx == state.size - 1:  # system in the end
        new_shape = (2 ** (state.size - 1), 2)
        new_axis = [1, 0]
        new_system = [which_system] + state.system[: system_idx]
    else:  # system in the middle
        new_shape = (2 ** system_idx, 2, 2 ** (state.size - system_idx - 1))
        new_axis = [1, 0, 2]
        new_system = [which_system] + state.system[: system_idx] + state.system[system_idx + 1:]
    new_vector = transpose(state.vector.reshape(new_shape), new_axis).reshape((state.length, 1))

    return State(new_vector, new_system)


def permute_systems(state, new_system):
    r""" Permute the quantum system to given order

    Assume that a quantum state :math:`\psi\rangle` can be decomposed to tensor product form:

    .. math::

        |\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes |\psi_3\rangle \otimes \cdots

    the labels of each :math:`|\psi_i\rangle` is :math:`i` , so the total labels of the current system are:

    .. math::

        \text{label} = \{1, 2, 3, \cdots \}

    the order of labels of the given new system is:

    .. math::

        \{i_1, i_2, i_3, \cdots \}

    The output new quantum state is:

    .. math::

        |\psi_{i_1}\rangle \otimes |\psi_{i_2}\rangle \otimes |\psi_{i_3}\rangle \otimes \cdots

    Args:
        state (State): the quantum state to be processed
        new_system (list): target order of the system

    Returns:
        State: the quantum state after permutation.
    """
    for label in reversed(new_system):
        state = permute_to_front(state, label)
    return state


class State:
    r"""Define the quantum state.

    Attributes:
        vector (Tensor): the column vector of the quantum state.
        system (list): the list of system labels of the quantum state.
    """

    def __init__(self, vector=None, system=None):
        r""" the constructor for initialize an object of the class `` "State`` .

        Args:
            vector (Tensor, optional): the column vector of the quantum state.
            system (list, optional): the list of system labels of the quantum state.
        """
        if vector is None and system is None:
            self.vector = array([1], dtype='float64')  # A trivial state
            self.system = []
            self.length = 1  # Length of state vector
            self.size = 0  # Number of qubits
        elif vector is not None and system is not None:
            assert vector.shape[0] >= 1 and vector.shape[1] == 1, "'vector' should be of shape [x, 1] with x >= 1."
            assert isinstance(system, list), "'system' should be a list."
            self.vector = vector
            self.system = system
            self.length = self.vector.shape[0]  # Length of state vector
            self.size = int(log2(self.length))  # Number of qubits
            assert self.size == len(self.system), "dimension of vector and system do not match."

        else:
            raise ValueError("we should either input both 'vector' and 'system' or input nothing.")
        self.state = [self.vector, self.system]

    def __str__(self):
        r"""print the information of this class
        """
        class_type_str = "State"
        vector_str = str(self.vector)
        system_str = str(self.system)
        length_str = str(self.length)
        size_str = str(self.size)
        print_str = class_type_str + "(" + \
                    "size=" + size_str + ", " + \
                    "system=" + system_str + ", " + \
                    "length=" + length_str + ", " + \
                    "vector=\r\n" + vector_str + ")"
        return print_str




