import numpy as np
import matplotlib.pyplot as plt
from Midterm2.Q2regression import multi_regress

def main():
    data = np.loadtxt("Question_2_DATA_rho_vp.txt")

    ln_vp = np.log(vp)     # Transform Vp to ln(Vp)

    p = data[:, 0]
    vp = data[:, 1]

    # Plot Vp vs rho
    plt.figure(figsize=(8, 5))
    plt.scatter(p, vp, marker = 'o')
    plt.xlabel("Density (g/cm³)")
    plt.ylabel("P-wave Velocity (m/s)")
    plt.title("P-wave Velocity vs. Density")
    plt.savefig("Q2_vp_vs_rho.png")
    plt.show()

if __name__ == "__main__":
    main()