import numpy as np
import matplotlib.pyplot as plt


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




