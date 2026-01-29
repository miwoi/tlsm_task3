import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax


class HybridCell(eqx.Module):

    
    # NN for the evolutionary equation 
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]
    
    #Physical Parameters
    E_infty: float
    E: float

    def __init__(self, E_infty: float, E: float, *, key: PRNGKeyArray):
        self.E_infty = E_infty
        self.E = E

        # Feed-Forward NN (MLP), learns f(eps, gamma) 
        # Input: 2 Dimensionen (eps, gamma)
        # Output: 1 Dimension (gamma_dot bzw. Rate der Änderung)
        self.layers = (
            klax.nn.Linear(2, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key), 
            klax.nn.Linear(16, 1, weight_init=he_normal(), key=key),
        )
        
        # activationfunction 
        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x, # Linearer Output für die Rate
        )

    def __call__(self, gamma, x):
        eps = x[0]
        dt = x[1]

        # Prediction of the rate of change by the NN
        # Input for the network: Current strain and current internal state
        nn_input = jnp.stack([eps, gamma])
        
        out = nn_input
        for layer, activation in zip(self.layers, self.activations):
            out = activation(layer(out))
        
         # The network returns the “speed” of the change (gamma_dot)
        gamma_dot = out[0]

        # 2. Updates the intern variable (Explicit Euler)
        # gamma_new = gamma_old + dt * gamma_dot
        gamma_new = gamma + dt * gamma_dot

        # Physical calculation of the stress (Hard-coded Physics)
        sig = self.E_infty * eps + self.E * (eps - gamma_new)

        return gamma_new, sig

class HybridModel(eqx.Module):
    cell: HybridCell

    def __init__(self, E_infty, E, *, key: PRNGKeyArray):

        self.cell = HybridCell(E_infty, E, key=key)

    def __call__(self, xs):
        def scan_fn(state, x):
            return self.cell(state, x)

        init_state = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_state, xs)

        return ys
