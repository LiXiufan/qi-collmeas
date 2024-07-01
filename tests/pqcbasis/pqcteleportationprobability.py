from jax.numpy import array, real, trace, log2
from backend.jax.qobject import X, Z
from backend.jax.simulator import depolarize, toProjector
from backend.jax.simulator import calculateFidelity
from backend.jax.simulator import operateMatrix, nKron, basisToObservable
from backend.jax.qobject import u3, I, CNOT, H, Ry, Rz
from backend.jax.routines import prepareBellStateMatrix

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

def QNNTwoCopy(params, densityMatrix, n):
    # Circuit handcoding
    ULayer1 = nKron([u3(params[0], params[1], params[2]), u3(params[3], params[4], params[5])] + [I()])

    HDecorateLayer21 = nKron([H(), H()] + [I()])
    CNOTLayer2 = nKron([CNOT()] + [I()])
    HDecorateLayer22 = nKron([H(), H()] + [I()])

    RyLayer3 = nKron([I(), Ry(params[6])] + [I()])

    CNOTLayer4 = nKron([CNOT()] + [I()])

    RzRyLayer5 = nKron([Rz(params[7]), Ry(params[8])] + [I()])

    HDecorateLayer61 = nKron([H(), H()] + [I()])
    CNOTLayer6 = nKron([CNOT()] + [I()])
    HDecorateLayer62 = nKron([H(), H()] + [I()])

    ULayer7 = nKron([u3(params[9], params[10], params[11]), u3(params[12], params[13], params[14])] + [I()])

    # ULayer1 = nKron(
    #     [u3(params[0], params[1], params[2]), u3(params[3], params[4], params[5])] + [I() for _ in range(n - 2)])
    #
    # HDecorateLayer21 = nKron([H(), H()] + [I() for _ in range(n - 2)])
    # CNOTLayer2 = nKron([CNOT()] + [I() for _ in range(n - 2)])
    # HDecorateLayer22 = nKron([H(), H()] + [I() for _ in range(n - 2)])
    #
    # RyLayer3 = nKron([I(), Ry(params[6])] + [I() for _ in range(n - 2)])
    #
    # CNOTLayer4 = nKron([CNOT()] + [I() for _ in range(n - 2)])
    #
    # RzRyLayer5 = nKron([Rz(params[7]), Ry(params[8])] + [I() for _ in range(n - 2)])
    #
    # HDecorateLayer61 = nKron([H(), H()] + [I() for _ in range(n - 2)])
    # CNOTLayer6 = nKron([CNOT()] + [I() for _ in range(n - 2)])
    # HDecorateLayer62 = nKron([H(), H()] + [I() for _ in range(n - 2)])
    #
    # ULayer7 = nKron(
    #     [u3(params[9], params[10], params[11]), u3(params[12], params[13], params[14])] + [I() for _ in range(n - 2)])


    gateList = [ULayer1, HDecorateLayer21, CNOTLayer2, HDecorateLayer22, RyLayer3, CNOTLayer4, RzRyLayer5, HDecorateLayer61, CNOTLayer6, HDecorateLayer62, ULayer7]

    # Operate martix
    for gate in gateList:
        densityMatrix = operateMatrix(densityMatrix, gate)

    return densityMatrix

def teleport(params, n, rho, beta):
    # Depolarized Bell state preparation
    # BellState = prepareBellStateMatrix()
    # BellStateMatrix = toProjector(BellState.getVector())
    BellStateMatrix = toProjector(array([[1], [0], [0], [1]]))
    depolarizedBellMatrix = depolarize(BellStateMatrix, beta)
    newRho = nKron([rho, depolarizedBellMatrix])

    # Ideal Bell measurement
    # newRho = operateMatrix(newRho, nKron([CNOT(), I()]))
    # newRho = operateMatrix(newRho, nKron([H(), I(), I()]))
    # Replace the above lines with twisted basis
    newRho = QNNTwoCopy(params, newRho, n + 2)

    stateAfter = measure(newRho)
    # outcome0, stateAfter = measure(newRho, 0)
    # outcome1, stateAfter = measure(stateAfter, 0)

    # Byproduct correction
    # if outcome0 == 1:
    #     stateAfter = operateMatrix(stateAfter, Z())
    # if outcome1 == 1:
    #     stateAfter = operateMatrix(stateAfter, X())
    # print("Teleported density matrix:", stateAfter)

    # Calculate the fidelity
    fidelity = calculateFidelity(rho, stateAfter)
    print("The fidelity becomes:", fidelity)
    return fidelity


























