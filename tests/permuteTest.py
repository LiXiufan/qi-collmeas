from qobject import State, zeroState, oneState
from qobject import CNOT
from simulator import permuteSystems, permuteToFront, nKron

dim = 3
Zeros = nKron([zeroState(), zeroState(), oneState()])
bg_state = State(vector=Zeros, system=['0', '1', '2'])
which_qubits = ['2', '0']

bg_state1 = permuteToFront(bg_state, which_qubits[1])
bg_state1 = permuteToFront(bg_state1, which_qubits[0])
new_state_len = bg_state1.getLength()
qua_length = int(new_state_len / 4)
cnot = CNOT()
# Reshape the state, apply CNOT and reshape it back

new_state_vector = (CNOT() @ bg_state1.getVector().reshape(4, qua_length)).reshape(new_state_len, 1)
new_state1 = State(vector=new_state_vector, system=bg_state1.getSystem())
print('new state 1', new_state1)

bg_state2 = permuteSystems(bg_state, ['2', '0', '1'])
print(bg_state2)

new_state_vector2 = (CNOT() @ bg_state2.getVector().reshape(4, qua_length)).reshape(new_state_len, 1)
new_state2 = State(vector=new_state_vector2, system=bg_state2.getSystem())
print('new state 2', new_state2)






