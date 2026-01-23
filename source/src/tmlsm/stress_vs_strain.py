import numpy as np
import matplotlib.pyplot as plt

""" This function shows the difference between Calibration and Test prediction 
    quality of the used models.

    Parameters of this function:
        a) Only one trained model should be used 
        b) Physic consistency / tendency to learn the physics of model
        is the main outcome of the figure
        c) Viscosity behavior and energy dissipation can be checked
        d) Used calibration (1 data) and test (1 data) load paths should be collected to understand 
        the different path effects

    The input of this function should be:
        1) Real stress and strain values for used to calibrate the model
        2) Predicted results of the model with calibration inputs
        3) Calculated Real stress and strain values of test data
        4) Predicted stress results of the model, which is trained with same calibration
         inputs, for test data strain and time
    """


def stress_strain_plot(cal_real_sig, cal_pred_sig, test_real_sig, test_pred_sig,
                       cal_real_eps, cal_pred_eps, test_real_eps, test_pred_eps):
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(cal_real_eps[0], cal_real_sig[0], color= "blue", label = "Real Value of Calibration")
    plt.plot(cal_pred_eps[0], cal_pred_sig[0], linestyle="--", color= "blue", label = "Predicted Result of Calibration")
    plt.plot(test_real_eps[0], test_real_sig[0], color = "red", label = "Real Value of Test" )
    plt.plot(test_pred_eps[0], test_pred_sig[0], linestyle="--", color= "red", label = "Predicted Value of Test")

    # Labels and title
    plt.xlabel("strain $\\varepsilon$")
    plt.ylabel("stress $\\sigma$")
    plt.title("Stress vs Strain Hysteresis Loops")
    plt.legend()
    
    # Grid
    plt.grid(True)
    plt.tight_layout()
    plt.show()





