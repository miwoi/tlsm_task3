"""Model implementations."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax


class Cell(eqx.Module):
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        self.layers = (
            klax.nn.Linear(3, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 2, weight_init=he_normal(), key=key),
        )

        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, gamma, x):
        eps = x[0]
        h = x[1]

        x = jnp.array([gamma, eps, h])

        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))

        print("This is the output: ", x)

        gamma = x[0]
        sig = x[1]

        return gamma, sig


class Model(eqx.Module):
    cell: Callable

    def __init__(self, *, key: PRNGKeyArray):
        self.cell = Cell(key=key)

    def __call__(self, xs):
        def scan_fn(state, x):
            return self.cell(state, x)

        init_state = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        return ys


def build(*, key: PRNGKeyArray):
    """Make and return a model instance."""
    return Model(key=key)







class HybridCell(eqx.Module):

    
    # Das neuronale Netz für die Evolutionsgleichung (trainierbar)
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]
    
    # Die physikalischen Parameter (fest, nicht trainierbar)
    E_infty: float
    E: float

    def __init__(self, E_infty: float, E: float, *, key: PRNGKeyArray):
        self.E_infty = E_infty
        self.E = E

        # Ein Feed-Forward NN (MLP), das die Funktion f(eps, gamma) lernt.
        # Input: 2 Dimensionen (eps, gamma)
        # Output: 1 Dimension (gamma_dot bzw. Rate der Änderung)
        self.layers = (
            klax.nn.Linear(2, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key), # evtl. keys splitten!
            klax.nn.Linear(16, 1, weight_init=he_normal(), key=key),
        )
        
        # Aktivierungsfunktionen
        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x, # Linearer Output für die Rate
        )

    def __call__(self, gamma, x):
        eps = x[0]
        dt = x[1]

        # 1. Vorhersage der Änderungsrate durch das NN
        # Input für das Netz: Aktuelle Dehnung und aktueller interner Zustand
        nn_input = jnp.stack([eps, gamma])
        
        out = nn_input
        for layer, activation in zip(self.layers, self.activations):
            out = activation(layer(out))
        
        # Das Netz gibt die "Geschwindigkeit" der Änderung zurück (gamma_dot)
        gamma_dot = out[0]

        # 2. Update der internen Variable (Expliziter Euler)
        # gamma_new = gamma_old + dt * gamma_dot
        gamma_new = gamma + dt * gamma_dot

        # 3. Physikalische Berechnung der Spannung (Hard-coded Physics)
        # Hier nutzen wir die bekannten Parameter E und E_infty
        sig = self.E_infty * eps + self.E * (eps - gamma_new)

        return gamma_new, sig

class HybridModel(eqx.Module):
    cell: HybridCell

    def __init__(self, E_infty, E, *, key: PRNGKeyArray):
        # Wir übergeben die festen Parameter und den Key für das NN
        self.cell = HybridCell(E_infty, E, key=key)

    def __call__(self, xs):
        def scan_fn(state, x):
            return self.cell(state, x)

        init_state = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        return ys