from qobject import zeroState, oneState, I, H, CNOT
from simulator import toProjector, nKron, basisToObservable, operateMatrix
from numpy import trace, matmul
from qutip import tensor

from simulator import measure

plusMatrix = toProjector(H() @ zeroState())
oneMatrix = toProjector(oneState())

outcome, state = measure(operateMatrix(nKron([plusMatrix, toProjector(zeroState())]), CNOT()), '0')
print(outcome)
print(state)

# TODO: 1. probability 2. Hardware

# rho = nKron([oneMatrix, zeroMatrix, zeroMatrix])
# obs = nKron([I(), basisToObservable(0), I()])
#
# a = trace(rho @ obs)
# print(a)
#
# b = (rho @ obs).reshape(2, 4, 2, 4)
# print(b)
# c = trace(b, axis1=0, axis2=2)
# print(c)














