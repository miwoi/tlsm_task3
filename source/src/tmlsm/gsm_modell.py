import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable

# 1. Das Energie-Netzwerk (Input: eps, gamma -> Output: Skalar e)
class EnergyNet(eqx.Module):
    layers: list
    activations: list

    def __init__(self, key):
        # Ein einfaches MLP (Multi-Layer Perceptron)
        # Input: 2 Dimensionen (eps, gamma)
        # Output: 1 Dimension (Energie e)
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(2, 16, key=keys[0]),
            eqx.nn.Linear(16, 16, key=keys[1]),
            eqx.nn.Linear(16, 1, key=keys[2])
        ]
        # Softplus ist gut für Energien (glatt und meist positiv), 
        # aber tanh/sigmoid ginge auch.
        self.activations = [jax.nn.softplus, jax.nn.softplus, lambda x: x]

    def __call__(self, eps, gamma):
        x = jnp.stack([eps, gamma])
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        # Rückgabe muss ein Skalar sein (kein Vektor der Länge 1)
        return x[0]

# 2. Die GSM-Zelle (Verbindet Physik und NN)
class GSMCell(eqx.Module):
    energy_net: EnergyNet
    g: float # Das inverse eta (1/eta)

    def __init__(self, eta, key):
        self.energy_net = EnergyNet(key=key)
        self.g = 1.0 / eta # g := eta^-1 = const

    def __call__(self, gamma, x):
        eps = x[0]
        dt = x[1]

        # --- Schritt A: Definition der Energie-Funktion für JAX ---
        # Wir definieren eine Hilfsfunktion, damit wir ableiten können.
        # JAX braucht eine Funktion f(eps, gamma) -> e
        def energy_fn(e_in, g_in):
            return self.energy_net(e_in, g_in)

        # --- Schritt B: Automatische Differenzierung ---
        # Wir berechnen die Gradienten der Energie nach eps und gamma.
        # grad(fn, argnums=(0, 1)) gibt Tupel (d_e/d_eps, d_e/d_gamma) zurück.
        d_energy_func = jax.grad(energy_fn, argnums=(0, 1))
        
        d_e_d_eps, d_e_d_gamma = d_energy_func(eps, gamma)

        # --- Schritt C: Physikalische Gesetze auswerten ---
        
        # 1. Spannung ist die Ableitung der Energie nach der Dehnung
        # sigma = d_e / d_eps
        sig = d_e_d_eps

        # 2. Entwicklung der internen Variable (Thermodynamische Konsistenz)
        # Die treibende Kraft ist -d_e / d_gamma.
        # Die Geschwindigkeit ist Kraft * Beweglichkeit (g).
        # gamma_dot = - g * (d_e / d_gamma)
        gamma_dot = -self.g * d_e_d_gamma

        # 3. Update (Expliziter Euler)
        gamma_new = gamma + dt * gamma_dot

        return gamma_new, sig

# 3. Das vollständige Modell (Wrapper für Zeitreihen)


class GSMModel(eqx.Module):
    cell: GSMCell

    def __init__(self, eta, key):
        self.cell = GSMCell(eta, key=key)

    def __call__(self, xs):
        def scan_fn(state, x):
            # state ist gamma, x ist (eps, dt)
            return self.cell(state, x)

        init_gamma = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_gamma, xs)
        return ys