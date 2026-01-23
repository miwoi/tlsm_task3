import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple

class MaxwellCell(eqx.Module):
  
    # Feste Materialparameter als Attribute
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

        # Update der internen Variable (Gamma)
        # Formel gamma_new = gamma + dt * E / eta * (eps - gamma)

        # Dies ist die Diskretisierung der DGL: d(gamma)/dt = (E/eta) * (eps - gamma)
        gamma_new = gamma + dt * (self.E / self.eta) * (eps - gamma)

        # Berechnung der Spannung (Sigma)
        # Formel aus data.py: sig = E_inf * eps + E * (eps - gamma_new)
        sig = self.E_infty * eps + self.E * (eps - gamma_new)

        # Rückgabe: Neuer Zustand (für den nächsten Schritt) und Output (für diesen Schritt)
        return gamma_new, sig


class MaxwellModel(eqx.Module):
   
    cell: MaxwellCell

    def __init__(self, E_infty: float, E: float, eta: float):
        # Initialisierung der Zelle mit den festen Parametern
        self.cell = MaxwellCell(E_infty, E, eta)

    def __call__(self, xs):
        """
        Wendet das Modell auf eine ganze Zeitreihe an.
        xs: Tupel aus (Dehnungen, Zeitschritte) -> (eps, dts)
        """
        # Scan-Funktion definieren, die den Zustand weiterträgt
        def scan_fn(state, x):
            return self.cell(state, x)

        # Initialer Zustand für gamma (Start bei 0)
        init_state = jnp.array(0.0)
        
        # jax.lax.scan ersetzt die Python for-Schleife aus data.py
        # Es geht effizient durch alle Zeitschritte in xs
        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        
        # ys enthält die gesammelten Spannungen (sig) aller Zeitschritte
        return ys