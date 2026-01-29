import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple



#### This model does not serve the purpose of the actual task. 

class MaxwellCell(eqx.Module):
  

    # material parameters 
    E_infty: float
    E: float
    eta: float

    def __init__(self, E_infty: float, E: float, eta: float):
        self.E_infty = E_infty
        self.E = E
        self.eta = eta

    def __call__(self, gamma: jnp.ndarray, x: Tuple[jnp.ndarray, jnp.ndarray]):
    
        eps = x[0]
        dt = x[1]

        # Update the intern variable (Gamma)
        # Formel gamma_new = gamma + dt * E / eta * (eps - gamma)
        # Discretization of the Ordinary DGL: d(gamma)/dt = (E/eta) * (eps - gamma)
        gamma_new = gamma + dt * (self.E / self.eta) * (eps - gamma)


        # Calculation of the stress 
        # Formel: sig = E_inf * eps + E * (eps - gamma_new)
        sig = self.E_infty * eps + self.E * (eps - gamma_new)

        #returns state for the next timestap and the stress for the current timestamp
        return gamma_new, sig


class MaxwellModel(eqx.Module):
   
    cell: MaxwellCell

    def __init__(self, E_infty: float, E: float, eta: float):
        #Initilize cell with given parameters 
        self.cell = MaxwellCell(E_infty, E, eta)

    def __call__(self, xs):
   
        # 
        def scan_fn(state, x):
            return self.cell(state, x)

        # Initial state für gamma (Start bei 0)
        init_state = jnp.array(0.0)

        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        
        # ys contains the collected stresses (sig) of all time steps
        return ys