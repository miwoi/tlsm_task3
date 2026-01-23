import datetime
import klax
import jax.random as jrandom
import time
import jax

<<<<<<< HEAD
import src.tmlsm.data as td
import src.tmlsm.plots as tp
import src.tmlsm.models as tm
=======
import tmlsm.data as td
import tmlsm.plots as tp
import tmlsm.models as tm
import tmlsm.maxwell_modell as mm

>>>>>>> e6b3c994077a75d99eb8e29804c831fb18830c2a

now = datetime.datetime.now


def main():
    # Load and visualize data
    E_infty = 0.5
    E = 2.0
    eta = 1.0
    n = 100
<<<<<<< HEAD
    omegas = [1.0,1.0]
    As = [1.0,5.0]
=======

    omegas = [1.0]
    As = [4.0]

    #RNN_Modell(E_infty, E, eta, n, omegas, As)

    #Maxwell_Modell(E_infty, E, eta, n, omegas, As)

    Hybrid_Modell(E_infty, E, eta, n, omegas, As)





def Hybrid_Modell(E_infty, E, eta, n, omegas, As):  
     
     # 1. Daten generieren (wie gehabt)
    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    
    # 2. Modell initialisieren
  
    key = jrandom.PRNGKey(time.time_ns())
    keys = jrandom.split(key, 2)



    model = tm.HybridModel(E_infty, E, key=keys[0])

    # 3. Training (Kalibrierung)
    
    t1 = now()
    print("Starte Training des Hybrid-Modells...")
    model, history = klax.fit(
        model,
        ((eps, dts), sig), # Input: (Dehnung, Zeit), Target: Spannung
        batch_axis=0,
        steps=2000, # Evtl. mehr Steps nötig
        key=key,
    )

    t2 = now()
    print(f"it took {(t2 - t1).total_seconds():.2f} (sec) to calibrate the model")

    history.plot()

    # Unwrap all wrappers and apply all constraints to the model
    model_ = klax.finalize(model)

    ### Plotting 
    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)

    As = [1, 1, 2]
    omegas = [1, 2, 3]

    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)

    eps, eps_dot, sig, dts = td.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)






def Maxwell_Modell(E_infty, E, eta, n, omegas, As): 
     

    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As) 
    new_maxwell_modell = mm.MaxwellModel(E_infty, E, eta)
    sig_pred = jax.vmap(new_maxwell_modell)((eps, dts))   
    tp.plot_model_pred(eps, sig, sig_pred, omegas, As)


   





def RNN_Modell(E_infty, E, eta, n, omegas, As): 

>>>>>>> e6b3c994077a75d99eb8e29804c831fb18830c2a

    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    print(eps.shape)
    tp.plot_data(eps, eps_dot, sig, omegas, As)

    # Create a random key for the random weight initialization and the
    # batch generation.
    key = jrandom.PRNGKey(time.time_ns())
    print("base keys",key)
    keys = jrandom.split(key, 2)

    # Build model instance
    model = tm.build(key=keys[0])

    ### Training 
    # Calibrate the model
    t1 = now()
    print(t1)

    model, history = klax.fit(
        model,
        ((eps, dts), sig),
        batch_axis=0,
        steps=10000,
        history=klax.HistoryCallback(log_every=1),
        key=keys[1],
    )

    t2 = now()
    print(f"it took {(t2 - t1).total_seconds():.2f} (sec) to calibrate the model")

    history.plot()

    # Unwrap all wrappers and apply all constraints to the model
    model_ = klax.finalize(model)

    ### Plotting 
    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)

    As = [1, 1, 2,3]
    omegas = [1, 2, 3,1]

    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)

    eps, eps_dot, sig, dts = td.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)


if __name__ == "__main__":
        main()
