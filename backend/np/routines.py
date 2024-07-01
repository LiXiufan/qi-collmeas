from backend.np.qobject import CNOT, H, zeroState
from backend.np.simulator import nKron, operateVector
from backend.np.qobject import State

def prepareBellStateMatrix():
    n = 2
    zeros = nKron([zeroState() for _ in range(n)])
    BellState = State(vector=zeros, system=[str(i) for i in range(n)])
    gateList = [[H(), ['0']], [CNOT(), ['0', '1']]]
    for gate in gateList:
        BellState = operateVector(BellState, gate)
    return BellState











