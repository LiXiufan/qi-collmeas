from qobject import ZeroState, OneState, PlusState, MinusState
from qobject import H, I, X, Z, Y
from qobject import Rx, Ry, Rz, u3
from qobject import Nkron

from qiskit.quantum_info import random_density_matrix, random_statevector

dim = 1
eps = 1e-12

# Pure state
# rho = random_statevector(2 ** dim).data.reshape(2 ** dim, 1)
# rho = rho @ transpose(conj(rho))


# Random state
rho = random_density_matrix(2 ** dim).data.reshape(2 ** dim, 2 ** dim)
print(rho)
