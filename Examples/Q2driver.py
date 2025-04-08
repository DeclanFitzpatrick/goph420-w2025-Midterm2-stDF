import numpy as np
import matplotlib.pyplot as plt
from Midterm2.Q2regression import multi_regress

def main():
    data = np.loadtxt("Question_2_DATA_rho_vp.txt")

    p = data[:, 0]
    vp = data[:, 1]

    # Plot Vp vs rho
    plt.figure(figsize=(8, 5))
    plt.scatter(p, vp, marker = 'o')
    plt.xlabel("Density (g/cm³)")
    plt.ylabel("P-wave Velocity (m/s)")
    plt.title("P-wave Velocity vs. Density")
    plt.savefig("figures/Q2_vp_vs_rho.png")
    plt.show()

    y = np.log(vp)

    Z = np.vstack((np.ones_like(p), p)).T  # design matrix

    a, e, rsq = multi_regress(y, Z)  # use least square regression to find coeffs, residuals, and R^2

    print(f'a: {a}')
    print(f' e: {e}')
    print(f' rsq: {rsq}')

    log_n = Z @ a  # compute predicted values

        # Plot ln(Vp) vs rho with regression line
    plt.figure(figsize=(8, 5))
    plt.scatter(p, y, label='Observed ln(Vp)', color='green')
    plt.plot(p, log_n, label='Fitted line', color='black')
    plt.xlabel("Density (g/cm³)")
    plt.ylabel("ln(P-wave Velocity)")
    plt.title("Linearized Fit: ln(Vp) vs. Density")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/Q2_lnvp_fit.png")
    plt.show()


if __name__ == "__main__":
    main()