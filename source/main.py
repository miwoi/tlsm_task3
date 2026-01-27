import datetime
import os
import klax
import jax.random as jrandom
import time
import jax
import matplotlib.pyplot as plt

import src.tmlsm.data as td
import src.tmlsm.models as tm
import src.tmlsm.plots.calibration_vs_test_in_time as p_cal_test
import src.tmlsm.plots.stress_vs_strain as p_stress_strain
import src.tmlsm.plots.interpol_vs_extrapol as p_inter_extra
import src.tmlsm.plots.plots as tp


now = datetime.datetime.now


def main():
    # --- Data Parameters ---
    E_infty = 0.5
    E = 2.0
    eta = 1.0
    n = 100
    NumberSteps = 10000

    


    # Calibration Data (Interpolation)
    omegas_cal = [1.0, 1.0]
    As_cal = [1.0, 5.0]
    eps_cal, eps_dot_cal, sig_cal, dts_cal = td.generate_data_harmonic(E_infty, E, eta, n, omegas_cal, As_cal)
    print(f"Calibration data shape: {eps_cal.shape}")

    # Test Data (Extrapolation)
    omegas_test = [1.0, 2.0, 3.0, 1.0]
    As_test = [1.0, 1.0, 2.0, 3.0]
    eps_test, eps_dot_test, sig_test, dts_test = td.generate_data_harmonic(E_infty, E, eta, n, omegas_test, As_test)

    # Test Data (Relaxation)
    eps_relax, eps_dot_relax, sig_relax, dts_relax = td.generate_data_relaxation(E_infty, E, eta, n, omegas_test, As_test)

    # --- Model Execution Loop ---
    base_key = jrandom.PRNGKey(time.time_ns())
    print("Base key:", base_key)



    # Define models to run
    models_to_run = ["RNN", "Hybrid"]

    for model_name in models_to_run:
        print(f"\n{'='*10} Running {model_name} Model {'='*10}")
        
        # Split keys for this run
        base_key, key_init, key_train = jrandom.split(base_key, 3)

        # Build model instance
        if model_name == "RNN":
            model = tm.build(key=key_init)
        elif model_name == "Hybrid":
            model = tm.HybridModel(E_infty, E, key=key_init)
        else:
            continue

        ### Training 
        t1 = now()
        model, history = klax.fit(
            model,
            ((eps_cal, dts_cal), sig_cal),
            batch_axis=0,
            steps=NumberSteps,
            history=klax.HistoryCallback(log_every=1),
            key=key_train,
        )
        t2 = now()
        print(f"Calibration took {(t2 - t1).total_seconds():.2f} seconds")

        # Unwrap model
        model_ = klax.finalize(model)

        ### Prediction
        sig_m_cal = jax.vmap(model_)((eps_cal, dts_cal))
        sig_m_test = jax.vmap(model_)((eps_test, dts_test))
        sig_m_relax = jax.vmap(model_)((eps_relax, dts_relax))

        ### Saving Plots
        timestamp = int(time.time())
        save_dir = os.path.join("created_plots", model_name, str(timestamp))
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving plots to: {save_dir}")

        # 1. Calibration vs Test (Time)
        p_cal_test.cal_test_time(
            sig_cal, sig_m_cal, 
            sig_test, sig_m_test, 
            save_path=os.path.join(save_dir, "calibration_vs_test_time.png")
        )

        # 2. Stress vs Strain
        # Note: Model predicts stress, so we use input eps for both real and pred x-axis
        p_stress_strain.stress_strain_plot(
            sig_cal, sig_m_cal, 
            sig_test, sig_m_test,
            eps_cal, eps_cal, 
            eps_test, eps_test,
            save_path=os.path.join(save_dir, "stress_vs_strain.png")
        )

        # 3. Interpolation vs Extrapolation
        p_inter_extra.interpol_extrapol(
            sig_cal, sig_m_cal, 
            sig_test, sig_m_test, 
            save_path=os.path.join(save_dir, "interpol_vs_extrapol.png")
        )

        # 4. Plots from plots.py (Data Calibration)
        tp.plot_data(eps_cal, eps_dot_cal, sig_cal, omegas_cal, As_cal, save_path=os.path.join(save_dir, "data_calibration.png"))

        # 5. Plots from plots.py (Model Prediction Calibration)
        tp.plot_model_pred(eps_cal, sig_cal, sig_m_cal, omegas_cal, As_cal, save_path=os.path.join(save_dir, "model_prediction_calibration.png"))

        # 6. Plots from plots.py (Data Test)
        tp.plot_data(eps_test, eps_dot_test, sig_test, omegas_test, As_test, save_path=os.path.join(save_dir, "data_test.png"))

        # 7. Plots from plots.py (Model Prediction Test)
        tp.plot_model_pred(eps_test, sig_test, sig_m_test, omegas_test, As_test, save_path=os.path.join(save_dir, "model_prediction_test.png"))

        # 8. Plots from plots.py (Data Relaxation)
        tp.plot_data(eps_relax, eps_dot_relax, sig_relax, omegas_test, As_test, save_path=os.path.join(save_dir, "data_relaxation.png"))

        # 9. Plots from plots.py (Model Prediction Relaxation)
        tp.plot_model_pred(eps_relax, sig_relax, sig_m_relax, omegas_test, As_test, save_path=os.path.join(save_dir, "model_prediction_relaxation.png"))

if __name__ == "__main__":
        main()
