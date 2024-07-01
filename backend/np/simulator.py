from numpy import kron, log2, trace
from numpy import transpose, conj, identity, real
from backend.np.qobject import State
from backend.np.qobject import I, Z
from numpy import random
from scipy.linalg import sqrtm

def toProjector(vector):
    return vector @ transpose(conj(vector))


def nKron(matrices):
    dim = len(matrices)
    if dim == 1:
        kronMatrix = matrices[0]
    else:
        kronMatrix = identity(1)
        for i in range(dim):
            kronMatrix = kron(kronMatrix, matrices[i])
    return kronMatrix


def permuteToFront(state, whichSystem):
    systemIdx = state.getSystem().index(whichSystem)
    if systemIdx == 0:  # system in the front
        return state
    elif systemIdx == state.getSize() - 1:  # system in the end
        newShape = (2 ** (state.getSize() - 1), 2)
        newAxis = [1, 0]
        newSystem = [whichSystem] + state.getSystem()[: systemIdx]
    else:  # system in the middle
        newShape = (2 ** systemIdx, 2, 2 ** (state.getSize() - systemIdx - 1))
        newAxis = [1, 0, 2]
        newSystem = [whichSystem] + state.getSystem()[: systemIdx] + state.getSystem()[systemIdx + 1:]
    newVector = transpose(state.getVector().reshape(newShape), newAxis).reshape((state.getLength(), 1))
    return State(newVector, newSystem)


def permuteSystems(state, newSystem):
    for label in reversed(newSystem):
        state = permuteToFront(state, label)
    return state


def depolarize(rho, beta):
    return (1 - beta) * rho + beta / int(rho.shape[0]) * identity(int(rho.shape[0]))


def operateVector(state, gate):
    gateMat = gate[0]
    gateSys = gate[1]

    if len(gateSys) == 1:
        operatorLength = 2
        system = gateSys[0]
        bgState = permuteToFront(state, system)
    else:
        operatorLength = 4
        cq = gateSys[0]
        tq = gateSys[1]
        bgState = permuteToFront(state, tq)
        bgState = permuteToFront(bgState, cq)

    newStateLen = bgState.getLength()
    permuteLength = int(newStateLen / operatorLength)
    # Reshape the state, apply the gate and reshape it back
    newStateVector = (gateMat @ bgState.getVector().reshape(operatorLength, permuteLength)).reshape(newStateLen, 1)
    newState = State(vector=newStateVector, system=bgState.getSystem())
    return newState


def operateMatrix(rho, u):
    return u @ rho @ transpose(conj(u))


def measure(rho, whichQubit):
    whichQubitInt = int(whichQubit)
    prob = [0, 0]
    stateAfterMeasurement = [0, 0]
    n = int(log2(rho.shape[0]))
    results = ['0', '1']
    for i, result in enumerate(results):
        basisM = int(result[0])
        obsM = nKron([I() for _ in range(whichQubitInt)] + [basisToObservable(basisM)] + [I() for _ in range(n - whichQubitInt - 1)])
        probResult = float(real(trace(obsM @ rho)))
        prob[i] = probResult
        stateAfter = (obsM @ rho).reshape(2, 2 ** (n-1), 2, 2 ** (n-1))
        stateAfterMeasurement[i] = trace(stateAfter, axis1=0, axis2=2) / probResult

    outcome = random.choice([0, 1], p=prob)
    return outcome, stateAfterMeasurement[outcome]


def basisToObservable(basisInt):
    if basisInt == 0:
        return (I() + Z()) / 2
    elif basisInt == 1:
        return (I() - Z()) / 2
    else:
        raise ValueError()

def calculateFidelity(rho1, rho2):
    return real(trace(sqrtm(sqrtm(rho1) @ rho2 @ sqrtm(rho1))) ** 2)















