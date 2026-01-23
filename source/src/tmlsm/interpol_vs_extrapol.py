import numpy as np
import matplotlib.pyplot as plt

# Yapilacak: Her plota isim ekle ki interpol - extrapol ayrimi olabilsin

def interpol_extrapol(cal_real_sig, cal_pred_sig, test_real_sig, test_pred_sig):
    
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
    plt.show()




