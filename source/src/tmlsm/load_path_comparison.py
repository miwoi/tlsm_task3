import numpy as np
import matplotlib.pyplot as plt


def load_path_effect(test_real_sig, diverse_path_sig, narrow_path_sig):

    # Time step
    n = len(test_real_sig[0])
    ns = np.linspace(0, 2 * np.pi, n)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(ns, test_real_sig[0], color= "blue", label = "Real Value of Test")
    plt.plot(ns, diverse_path_sig[0], linestyle="--", color= "blue", label = "Predicted Value wrt Diverse Load Path")
    plt.plot(ns, narrow_path_sig[0], color = "red", label = "Real Value wrt Narrow Load Path" )

    # Labels and title
    plt.xlabel("time $t$")
    plt.ylabel("stress $\\sigma$")
    plt.title("Effect of Load Path Choice (Generalization)")
    plt.legend()
    
    # Grid
    plt.grid(True)
    plt.tight_layout()
    plt.show()