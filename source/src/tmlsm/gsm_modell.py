import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable

#Energy-Network (Input: eps, gamma -> Output: Skalar e)
class EnergyNet(eqx.Module):
    layers: list
    activations: list

    def __init__(self, key):


        # (Multi-Layer Perceptron)
        # Input: 2 Dimensions (eps, gamma)
        # Output: 1 Dimension (Energie e)
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(2, 16, key=keys[0]),
            eqx.nn.Linear(16, 16, key=keys[1]),
            eqx.nn.Linear(16, 1, key=keys[2])
        ]
       
        self.activations = [jax.nn.softplus, jax.nn.softplus, lambda x: x]

    def __call__(self, eps, gamma):
        x = jnp.stack([eps, gamma])
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))

       # Output has to be a scalar 

        return x[0]

#GSM cell 
class GSMCell(eqx.Module):
    energy_net: EnergyNet

    g: float # Inverse eta (1/eta)

    def __init__(self, eta, key):
        self.energy_net = EnergyNet(key=key)
        self.g = 1.0 / eta # g := eta^-1 = const

    def __call__(self, gamma, x):
        eps = x[0]
        dt = x[1]

     
        #definiton of the help funcion (energy function)
        #For Jax f(eps, gamma) -> e

        def energy_fn(e_in, g_in):
            return self.energy_net(e_in, g_in)


        #calculation of the derivative 
        #gradient of the energy for eps and gamma 
        # grad(fn, argnums=(0, 1)) returns  (d_e/d_eps, d_e/d_gamma)
        d_energy_func = jax.grad(energy_fn, argnums=(0, 1))
        d_e_d_eps, d_e_d_gamma = d_energy_func(eps, gamma)


        # Evaluate physics 
        # stress as the derivitave of energy with resepct to strain 
        # sigma = d_e / d_eps

        sig = d_e_d_eps

        # Development of the internal variable (thermodynamic consistency)
        # The driving force is -d_e / d_gamma.
        gamma_dot = -self.g * d_e_d_gamma

        #Update (Expliziter Euler)
        gamma_new = gamma + dt * gamma_dot

        return gamma_new, sig



#complete modell

class GSMModel(eqx.Module):
    cell: GSMCell

    def __init__(self, eta, key):
        self.cell = GSMCell(eta, key=key)

    def __call__(self, xs):
        def scan_fn(state, x):
        
            return self.cell(state, x)

        init_gamma = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_gamma, xs)
        return ys