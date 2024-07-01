from backend.jax.qobject import X, Z
from jax.numpy import array
from backend.jax.simulator import depolarize, toProjector
from backend.jax.simulator import measure, calculateFidelity
from backend.jax.simulator import operateMatrix, nKron
from backend.jax.qobject import u3, I, CNOT, H, Ry, Rz
from backend.jax.routines import prepareBellStateMatrix

def QNNTwoCopy(params, densityMatrix, n):
    # Circuit handcoding
    # ULayer1 = nKron([u3(params[0], params[1], params[2]), u3(params[3], params[4], params[5])] + [I() for _ in range(n-2)])
    #
    # HDecorateLayer21 = nKron([H(), H()] + [I() for _ in range(n-2)])
    # CNOTLayer2 = nKron([CNOT()] + [I() for _ in range(n-2)])
    # HDecorateLayer22 = nKron([H(), H()] + [I() for _ in range(n-2)])
    #
    # RyLayer3 = nKron([I(), Ry(params[6])] + [I() for _ in range(n-2)])
    #
    # CNOTLayer4 = nKron([CNOT()] + [I() for _ in range(n-2)])
    #
    # RzRyLayer5 = nKron([Rz(params[7]), Ry(params[8])] + [I() for _ in range(n-2)])
    #
    # HDecorateLayer61 = nKron([H(), H()] + [I() for _ in range(n-2)])
    # CNOTLayer6 = nKron([CNOT()] + [I() for _ in range(n-2)])
    # HDecorateLayer62 = nKron([H(), H()] + [I() for _ in range(n-2)])
    #
    # ULayer7 = nKron([u3(params[9], params[10], params[11]), u3(params[12], params[13], params[14])] + [I() for _ in range(n-2)])

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


    gateList = [ULayer1, HDecorateLayer21, CNOTLayer2, HDecorateLayer22, RyLayer3, CNOTLayer4, RzRyLayer5, HDecorateLayer61, CNOTLayer6, HDecorateLayer62, ULayer7]

    # Operate martix
    for gate in gateList:
        densityMatrix = operateMatrix(densityMatrix, gate)

    return densityMatrix

def teleport(params, n, rho, beta):
    # Depolarized Bell state preparation
    # BellState = prepareBellStateMatrix()
    # BellState = toProjector()
    # BellStateMatrix = toProjector(BellState.getVector())
    BellStateMatrix = toProjector(array([[1], [0], [0], [1]]))
    depolarizedBellMatrix = depolarize(BellStateMatrix, beta)
    newRho = nKron([rho, depolarizedBellMatrix])

    # Ideal Bell measurement
    # newRho = operateMatrix(newRho, nKron([CNOT(), I()]))
    # newRho = operateMatrix(newRho, nKron([H(), I(), I()]))
    # Replace the above lines with twisted basis
    newRho = QNNTwoCopy(params, newRho, n + 2)
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
    return fidelity


























