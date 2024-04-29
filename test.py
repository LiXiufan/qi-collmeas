from qobject import I, zeroState, CNOT, H, X, Z, Rz
from simulator import depolarize, toProjector, nKron
from qiskit.quantum_info import random_density_matrix, random_statevector
from routines import prepareBellStateMatrix
from simulator import operateMatrix, measure
from numpy import trace

n = 1
dim = 2 ** n
# beta = 0.1
beta = 0
rho = random_density_matrix(dim).data
print(rho)

BellState = prepareBellStateMatrix()
BellStateMatrix = toProjector(BellState.getVector())
depolarizedBellMatrix = depolarize(BellStateMatrix, beta)
newRho = nKron([rho, depolarizedBellMatrix])

newRho = operateMatrix(newRho, nKron([CNOT(), I()]))
newRho = operateMatrix(newRho, nKron([H(), I(), I()]))
# rhoIdeal =

# PQC here
# def PQC(rho, params):
#     rho = operateMatrix(rho, Rz(params[0]))
#
#
#     return rho
#
#
# def costFunction(rho,):
#     trace(rho, rhoIdeal)
#
#
# def optimize():


# TODO: 1. figure, fidelity improvement with collective mea 2. analyze collective measurements from theoretical pers.
#  3. two different channels  4. change PQC, learning theory, 5. applications: quantum networks, MBQC, improved fidelity









#
#
# outcome0, stateAfter = measure(newRho, 0)
# outcome1, stateAfter = measure(stateAfter, 0)
# print(outcome0)
# print(outcome1)
#
# if outcome0 == 1:
#     stateAfter = operateMatrix(stateAfter, Z())
#
# if outcome1 == 1:
#     stateAfter = operateMatrix(stateAfter, X())
#
#
# print(stateAfter)












