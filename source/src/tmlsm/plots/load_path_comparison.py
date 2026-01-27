import numpy as np
import matplotlib.pyplot as plt

""" This function shows the importance of choosing correct load path for train the model.

    Parameters of this function:
        a) Two trained model with different calibration data should be used
        b) The trained model should be tested with same test data
        c) Dataset design effect is the main outcome of this figure
        d) Used calibration (2 data) and test (1 data) load paths should be collected to understand 
        the different path effects

    The input of this function should be:
        1) Real stress values for used to calibrate both models
        2) Predicted stress test results of the initial model
        3) Predicted stress test results of the second model
    """

def load_path_effect(test_real_sig, diverse_path_sig, narrow_path_sig):

    # Time step
    n = len(test_real_sig[0])
    ns = np.linspace(0, 2 * np.pi, n)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(ns, test_real_sig[0], color= "blue", label = "Real Value of Test")
    plt.plot(ns, diverse_path_sig[0], linestyle="--", color= "blue", label = "Predicted Value wrt Diverse Load Path")
    plt.plot(ns, narrow_path_sig[0], color = "red", label = "Predicted Value wrt Narrow Load Path" )

    # Labels and title
    plt.xlabel("time $t$")
    plt.ylabel("stress $\\sigma$")
    plt.title("Effect of Load Path Choice (Generalization)")
    plt.legend()
    
    # Grid
    plt.grid(True)
    plt.tight_layout()
    plt.show()