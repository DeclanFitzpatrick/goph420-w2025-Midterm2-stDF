import numpy as np
import matplotlib.pyplot as plt
from Midterm2.Q2regression import multi_regress

def main():
    data = np.loadtxt("Question_2_DATA_rho_vp.txt")

    p = data[:, 0]  #density data
    vp = data[:, 1] # P-wave velocity data

    # Plot Vp vs rho
    plt.figure(figsize=(8, 5))
    plt.scatter(p, vp, marker = 'o')
    plt.xlabel("Density (g/cm³)")
    plt.ylabel("P-wave Velocity (m/s)")
    plt.title("P-wave Velocity vs. Density")
    plt.savefig("figures/Raw_data.png")
    plt.show()

    # Part b)
    # Linearize the data using natural logarithm
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
    plt.text(0.20,0.60, f'$R^2$ = {rsq:8f}' , transform=plt.gca().transAxes,fontsize=12, bbox=dict(facecolor='white'))
    plt.savefig("figures/Q2_lnvp_fit.png")
    plt.show()

    # Part c)
    # Recover model parameters
    V0 = np.exp(a[0])   # V₀ = e^(intercept)
    k = a[1]             # slope is k

    print(f"\nRecovered model parameters:")
    print(f"V₀ (m/s): {V0:.2f}")
    print(f"k (cm³/g): {k:.4f}")

    print(f"\nFinal model: Vp = {V0:.2f} * exp({k:.4f} * rho)")

    # Part (d) - Model Evaluation
    # R^2 interpretation
    print(f"\nR² = {rsq:.4f}")
    if rsq > 0.9:
        print("The model explains a very high proportion of the variance — great fit!")
    elif rsq > 0.75:
        print("The model fits well, but there may be some structure not captured.")
    else:
        print("The model might be missing key features or non-linear effects.")


if __name__ == "__main__":
    main()