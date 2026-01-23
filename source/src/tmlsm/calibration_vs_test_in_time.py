import numpy as np
import matplotlib.pyplot as plt

""" This function shows the difference between Calibration and Test prediction 
    quality of the used models.

    Parameters of this function:
        a) Only one trained model should be used 
        b) Model accuracy is the main outcome of the figure
        c) Interpolation and extrapolation behavior can be checked
        d) Used calibration (1 data) and test (1 data) load paths should be collected to understand 
        the different path effects

    The input of this function should be:
        1) Real stress values for used to calibrate the model
        2) Predicted results of the model with calibration inputs
        3) Calculated Real stress values of test data
        4) Predicted stress results of the model, which is trained with same calibration
         inputs, for test data strain and time
    """

def cal_test_time(cal_real_sig, cal_pred_sig, test_real_sig, test_pred_sig):
    
    # Time step
    n = len(cal_real_sig[0])
    ns = np.linspace(0, 2 * np.pi, n)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(ns, cal_real_sig[0], color= "blue", label = "Real Value of Calibration")
    plt.plot(ns, cal_pred_sig[0], linestyle="--", color= "blue", label = "Predicted Value of Calibration")
    plt.plot(ns, test_real_sig[0], color = "red", label = "Real Value of Test" )
    plt.plot(ns, test_pred_sig[0], linestyle="--", color= "red", label = "Predicted Value of Test")

    # Labels and title
    plt.xlabel("time $t$")
    plt.ylabel("stress $\\sigma$")
    plt.title("Calibration vs Test (Stress vs Time)")
    plt.legend()
    
    # Grid
    plt.grid(True)
    plt.tight_layout()
    plt.show()




