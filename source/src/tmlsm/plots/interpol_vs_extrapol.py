import numpy as np
import matplotlib.pyplot as plt

# Yapilacak: Her plota isim ekle ki interpol - extrapol ayrimi olabilsin
""" This function shows the difference between Calibration and Test prediction 
    quality of the used models.

    Parameters of this function:
        a) Two trained model with different calibration data should be used
        b) Interpolation and extrapolation accuracy is the main outcome of the figure
        c) The efficiency for different (A,w) values can be understood
        d) Used calibration (2 data) and test (2 data) load paths should be collected to understand 
        the different path effects

    The input of this function should be:
        1) Real stress values for used to calibrate initial model
        2) Real stress values for used to calibrate second model
        3) Predicted stress test results of the initial model
        4) Predicted stress test results of the second model
    """

def interpol_extrapol(cal_real_sig, cal_pred_sig, test_real_sig, test_pred_sig, save_path=None):
    
    # Time step
    n = len(cal_real_sig[0])
    ns = np.linspace(0, 2 * np.pi, n)
    
    fig, axs = plt.subplots(2, figsize=(12, 7))
    fig.suptitle("Interpolation vs Extrapolation (Stress vs Time)")

    # Plot 1 
    ax = axs[0]
    ax.plot(ns, cal_real_sig[0], color= "blue", label = "Real Value for Interpolation Case")
    ax.plot(ns, cal_pred_sig[0], linestyle="--", color= "blue", label = "Predicted Value for Interpolation Case")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("stress $\\sigma$")


    # Plot 2
    ax = axs[1]
    ax.plot(ns, test_real_sig[0], color = "red", label = "Real Value for Extrapolation Case" )
    ax.plot(ns, test_pred_sig[0], linestyle="--", color= "red", label = "Predicted Value forExtrapolation Case")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("stress $\\sigma$")
    
    # Grid
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
