from backend.np.qobject import I, CNOT, H, X, Z
from backend.np.simulator import depolarize, toProjector, nKron
from qiskit.quantum_info import random_density_matrix
from backend.np.routines import prepareBellStateMatrix
from backend.np.simulator import operateMatrix, measure, calculateFidelity

# Initialize parameters
n = 1                                         # qubit number
dim = 2 ** n                                  # dimension
beta = 0.8                                      # depolarizing parameter
rho = random_density_matrix(dim).data         # random input density matrix
print("Random input density matrix:", rho)

# Depolarized Bell state preparation
BellState = prepareBellStateMatrix()
BellStateMatrix = toProjector(BellState.getVector())
depolarizedBellMatrix = depolarize(BellStateMatrix, beta)
newRho = nKron([rho, depolarizedBellMatrix])

# Ideal Bell measurement
newRho = operateMatrix(newRho, nKron([CNOT(), I()]))
newRho = operateMatrix(newRho, nKron([H(), I(), I()]))
outcome0, stateAfter = measure(newRho, 0)
outcome1, stateAfter = measure(stateAfter, 0)

# Byproduct correction
if outcome0 == 1:
    stateAfter = operateMatrix(stateAfter, Z())
if outcome1 == 1:
    stateAfter = operateMatrix(stateAfter, X())
print("Teleported density matrix:", stateAfter)










# Calculate the fidelity
fidelity = calculateFidelity(rho, stateAfter)
print("The fidelity becomes:", fidelity)








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












