from jax.numpy import array, identity
from jax.numpy import cos, sin
from jax.numpy import sqrt, log2

eps = 1e-12


def zeroState():
    return array([1, 0]).reshape(-1, 1)


def oneState():
    return array([0, 1]).reshape(-1, 1)


def plusState():
    return 1 / sqrt(2) * array([1, 1]).reshape(-1, 1)


def minusState():
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


def u3(theta, phi, lam):
    # Construct an arbitrary single qubit unitary
    return Rz(lam) @ Ry(phi) @ Rz(theta)


def CNOT():
    return array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class State:
    def __init__(self, vector=None, system=None):
        if vector is None and system is None:
            raise ValueError("Please input vector and system.")
        else:
            assert vector.shape[0] >= 1 and vector.shape[1] == 1, "'vector' should be of shape [x, 1] with x >= 1."
            assert isinstance(system, list), "'system' should be a list."
            self.__vector = vector
            self.__system = system
            self.__length = self.__vector.shape[0]  # Length of state vector
            self.__size = array(log2(self.__length), int)  # Number of qubits
            assert self.__size == len(self.__system), "dimension of vector and system do not match."
        self.__state = [self.__vector, self.__system]

    def getSize(self):
        return self.__size
    def getLength(self):
        return self.__length

    def getVector(self):
        return self.__vector

    def getSystem(self):
        return self.__system

    def getState(self):
        return self.__state




















