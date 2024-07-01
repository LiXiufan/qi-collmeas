from backend.np.qobject import I, CNOT, H, X, Z
# from backend.np.simulator import
from numpy import kron, log2, trace, array
from numpy import transpose, conj, identity, real
from backend.np.simulator import depolarize, toProjector, nKron, basisToObservable
from qiskit.quantum_info import random_density_matrix
from backend.np.routines import prepareBellStateMatrix
from backend.np.simulator import operateMatrix, calculateFidelity

def measure(rho):
    prob = [0, 0, 0, 0]
    stateAfterMeasurement = [0, 0, 0, 0]
    n = array(log2(rho.shape[0]), int)
    results = ['00', '01', '10', '11']
    for i, result in enumerate(results):
        basisM1 = int(result[0])
        basisM2 = int(result[1])
        obsM1 = nKron([basisToObservable(basisM1)])
        obsM2 = nKron([basisToObservable(basisM2)])
        obsM = nKron([obsM1, obsM2, I()])
        probResult = array(real(trace(obsM @ rho)), float)
        prob[i] = probResult
        stateAfter = (obsM @ rho).reshape(4, 2 ** (n-2), 4, 2 ** (n-2))
        stateAfterMeasurement[i] = trace(stateAfter, axis1=0, axis2=2) / probResult

    stateAfter = 0
    for i, result in enumerate(results):
        outcome0 = int(result[0])
        outcome1 = int(result[1])
        if outcome0 == 1:
            gate1 = Z()
            # stateAfter = operateMatrix(stateAfter, Z())
        else:
            gate1 = I()
        if outcome1 == 1:
            gate2 = X()
        else:
            gate2 = I()
        stateAfter += prob[i] * operateMatrix(operateMatrix(stateAfterMeasurement[i], gate1), gate2)

    # stateAfter = prob[0] * stateAfterMeasurement[0] + prob[1] * stateAfterMeasurement[1] +
    return stateAfter
    # return prob, stateAfterMeasurement



# Initialize parameters
n = 1                                         # qubit number
dim = 2 ** n                                  # dimension
beta = 0                                      # depolarizing parameter
# beta = 0.5                                      # depolarizing parameter
# beta = 0.8                                      # depolarizing parameter
rho = random_density_matrix(dim).data         # random input density matrix
print("Random input density matrix:", rho)

# Depolarized Bell state preparation
# BellState = prepareBellStateMatrix()
# BellStateMatrix = toProjector(BellState.getVector())
BellStateMatrix = toProjector(array([[1], [0], [0], [1]]))

depolarizedBellMatrix = depolarize(BellStateMatrix, beta)
newRho = nKron([rho, depolarizedBellMatrix])

# Ideal Bell measurement
newRho = operateMatrix(newRho, nKron([CNOT(), I()]))
newRho = operateMatrix(newRho, nKron([H(), I(), I()]))



# stateAfter = measure(newRho, 0)
# stateAfter = measure(stateAfter, 0)


# def measure(rho, whichQubit):
#     whichQubitInt = int(whichQubit)
#     prob = [0, 0]
#     stateAfterMeasurement = [0, 0]
#     n = array(log2(rho.shape[0]), int)
#     results = ['0', '1']
#     for i, result in enumerate(results):
#         basisM = array(result[0], int)
#         obsM = nKron([I() for _ in range(whichQubitInt)] + [basisToObservable(basisM)] + [I() for _ in range(n - whichQubitInt - 1)])
#         probResult = array(real(trace(obsM @ rho)), float)
#         prob[i] = probResult
#         stateAfter = (obsM @ rho).reshape(2, 2 ** (n-1), 2, 2 ** (n-1))
#         stateAfterMeasurement[i] = trace(stateAfter, axis1=0, axis2=2) / probResult
#     stateAfterProbability = prob[0] * stateAfterMeasurement[0] + prob[1] * stateAfterMeasurement[1]
#     return stateAfterProbability
#     # return prob, stateAfterMeasurement

# stateAfter = measure(newRho, 0)
# stateAfter = measure(stateAfter, 0)


    # outcome = random.choice([0, 1], p=prob)
    # return outcome, stateAfterMeasurement[outcome]



# outcome0, stateAfter = measure(newRho, 0)
# outcome1, stateAfter = measure(stateAfter, 0)

# Byproduct correction
# if outcome0 == 1:
#     stateAfter = operateMatrix(stateAfter, Z())
# if outcome1 == 1:
#     stateAfter = operateMatrix(stateAfter, X())
# print("Teleported density matrix:", stateAfter)
#
# Calculate the fidelity
stateAfter = measure(newRho)
print("The fidelity becomes:", calculateFidelity(rho, stateAfter))








