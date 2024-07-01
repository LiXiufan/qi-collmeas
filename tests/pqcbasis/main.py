from tests.pqcbasis.pqcteleportationprobability import teleport
import jax.numpy as jnp
import jax
import optax
from qiskit.quantum_info import random_density_matrix

# Initialize parameters
n = 1                                         # qubit number
dim = 2 ** n                                  # dimension
beta = 0                                      # depolarizing parameter
# beta = 0.5                                  # depolarizing parameter
# beta = 0.8                                  # depolarizing parameter
rho = random_density_matrix(dim).data         # random input density matrix
print("Random input density matrix:", rho)
params = jnp.array([0.0 for _ in range(15)])    # trainable parameters
eta = 0.1                                     # learning rate
ITR = 5                                       # iteration


# Use gradient descent and adam optimizer
def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    optInit = optimizer.init(params)

    @jax.jit
    def step(params, optInit, n, rho, beta):
        fidelity, grads = jax.value_and_grad(teleport)(params, n, rho, beta)
        updates, optInit = optimizer.update(grads, optInit, params)
        params = optax.apply_updates(params, updates)
        return params, optInit, fidelity

    for itr in range(ITR):
        params, optInit, fidelity = step(params, optInit, n, rho, beta)
        # if itr % 4 == 0:
        #     print(f'step {itr}, loss: {fidelity}')
        print(f'step {itr}, loss: {fidelity}')
    return params


# Finally, we can fit our parametrized function using the Adam optimizer
# provided by optax.
optimizer = optax.adam(learning_rate=eta)
params = fit(params, optimizer)
print(params)

