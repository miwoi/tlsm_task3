import numpy as np
import matplotlib.pyplot as plt

""" This function shows the time discretization sensitivity of the models.

    Parameters of this function:
        a) Three trained model with same calibration data should be used
        b) Three different time step increment should be determined and noted
        c) Time step effect on the model training is the main outcome of the figure
        d) Used calibration (1 data) and test (1 data) load paths 

    The input of this function should be:
        1) Predicted stress test results wrt first determined time step
        2) Predicted stress test results wrt second determined time step
        3) Predicted stress test results wrt third determined time step
    """

def time_step_plot(sig_ts_1, sig_ts_2, sig_ts_3):

    # Time step
    n = len(sig_ts_1[0])
    ns = np.linspace(0, 2 * np.pi, n)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(ns, sig_ts_1[0], color= "blue", label = "Time Step = 123")
    plt.plot(ns, sig_ts_2[0], color= "green", label = "Time Step = 123")
    plt.plot(ns, sig_ts_3[0], color = "red", label = "Time Step = 123" )

    # Labels and title
    plt.xlabel("time $t$")
    plt.ylabel("stress $\\sigma$")
    plt.title("Time Step Sensitivity)")
    plt.legend()
    
    # Grid
    plt.grid(True)
    plt.tight_layout()
    plt.show()

