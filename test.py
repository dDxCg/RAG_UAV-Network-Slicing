import numpy as np

class AlHouraniLoSModel:
    """
    Calculate LoS probability using Al-Hourani's environment-dependent model
    Based on the polynomial surface fitting approach
    """
    
    def __init__(self):
        # Table I - Coefficients for parameter 'a'
        self.C_a = np.array([
            [9.34e-01,  2.30e-01, -2.25e-03,  1.86e-05],  # j=0
            [1.97e-02,  2.44e-03,  6.58e-06,  0],          # j=1
            [-1.24e-04, -3.54e-06, 0,         0],          # j=2
            [2.73e-07,  0,         0,         0]           # j=3
        ])
        
        # Table II - Coefficients for parameter 'b'
        self.C_b = np.array([
            [1.17e+00, -7.56e-02,  1.98e-03, -1.78e-05],  # j=0
            [-5.79e-03, 1.81e-04, -1.65e-06,  0],         # j=1
            [1.73e-05, -2.02e-07,  0,         0],         # j=2
            [-2.00e-08, 0,         0,         0]          # j=3
        ])
    
    def calculate_s_curve_parameters(self, alpha, beta, gamma):
        """
        Calculate S-curve parameters a and b from environment parameters
        
        Args:
            alpha: Environment parameter (e.g., ratio of built-up land area)
            beta: Environment parameter (e.g., mean building height)
            gamma: Environment parameter (e.g., mean building density)
        
        Returns:
            a, b: S-curve parameters
        """
        # Calculate (alpha * beta) once
        alpha_beta = alpha * beta
        
        # Calculate parameter 'a' using polynomial
        a = self._calculate_polynomial(self.C_a, alpha_beta, gamma)
        
        # Calculate parameter 'b' using polynomial
        b = self._calculate_polynomial(self.C_b, alpha_beta, gamma)
        
        return a, b
    
    def _calculate_polynomial(self, C, alpha_beta, gamma):
        """
        Calculate polynomial: z = Σ Σ C_ij (αβ)^i γ^j
        
        Args:
            C: Coefficient matrix (4x4)
            alpha_beta: Product of α and β
            gamma: Environment parameter γ
        
        Returns:
            z: Calculated parameter (a or b)
        """
        z = 0.0
        
        # Sum over j from 0 to 3
        for j in range(4):
            gamma_power = gamma ** j
            
            # Sum over i from 0 to (3-j)
            for i in range(4 - j):
                alpha_beta_power = alpha_beta ** i
                z += C[i, j] * alpha_beta_power * gamma_power
        
        return z
    
    def p_los(self, theta_deg, alpha, beta, gamma):
        """
        Calculate LoS probability for given elevation angle
        
        Args:
            theta_deg: Elevation angle in degrees
            alpha, beta, gamma: Environment parameters
        
        Returns:
            P_LoS: Line-of-sight probability [0, 1]
        """
        # Get S-curve parameters
        a, b = self.calculate_s_curve_parameters(alpha, beta, gamma)
        
        # Calculate LoS probability using modified sigmoid
        # P(LoS, θ) = 1 / (1 + a·exp(-b[θ - a]))
        exponent = -b * (theta_deg - a)
        p_los = 1.0 / (1.0 + a * np.exp(exponent))
        
        return p_los
    
    def plot_los_curve(self, alpha, beta, gamma, theta_max=90):
        """
        Plot LoS probability curve for given environment
        """
        import matplotlib.pyplot as plt
        
        # Calculate parameters
        a, b = self.calculate_s_curve_parameters(alpha, beta, gamma)
        
        # Generate elevation angles
        theta = np.linspace(0, theta_max, 500)
        p = self.p_los(theta, alpha, beta, gamma)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(theta, p, 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Elevation Angle θ (degrees)', fontsize=12)
        plt.ylabel('LoS Probability', fontsize=12)
        plt.title(f'LoS Probability (α={alpha}, β={beta}, γ={gamma})\na={a:.3f}, b={b:.3f}', 
                  fontsize=14)
        plt.xlim([0, theta_max])
        plt.ylim([0, 1.05])
        
        # Mark key points
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='P=0.5')
        
        # Find angle at 50% LoS
        idx_50 = np.argmin(np.abs(p - 0.5))
        theta_50 = theta[idx_50]
        plt.axvline(x=theta_50, color='red', linestyle='--', alpha=0.5, 
                   label=f'θ_50={theta_50:.1f}°')
        
        plt.legend()
        plt.tight_layout()
        plt.show()


# Example usage with standard environments
def main():
    model = AlHouraniLoSModel()
    
    # Define environment parameters (examples from the paper)
    environments = {
        'suburban': {'alpha': 0.1, 'beta': 750, 'gamma': 8},
        'urban': {'alpha': 0.3, 'beta': 500, 'gamma': 15},
        'dense_urban': {'alpha': 0.5, 'beta': 300, 'gamma': 20},
        'high_rise': {'alpha': 0.3, 'beta': 1500, 'gamma': 10}
    }
    
    print("=" * 70)
    print("AL-HOURANI LoS MODEL - S-CURVE PARAMETERS")
    print("=" * 70)
    
    for env_name, params in environments.items():
        alpha = params['alpha']
        beta = params['beta']
        gamma = params['gamma']
        
        a, b = model.calculate_s_curve_parameters(alpha, beta, gamma)
        
        print(f"\n{env_name.upper()}:")
        print(f"  Environment: α={alpha}, β={beta}, γ={gamma}")
        print(f"  S-curve params: a={a:.4f}, b={b:.4f}")
        
        # Test at specific angles
        print(f"  LoS Probability:")
        for angle in [0, 10, 20, 30, 45, 60]:
            p = model.p_los(angle, alpha, beta, gamma)
            print(f"    θ={angle:2d}°: {p:.3f}")
    
    print("\n" + "=" * 70)
    
    # Plot for suburban environment
    print("\nPlotting suburban environment curve...")
    model.plot_los_curve(
        alpha=0.3, 
        beta=500, 
        gamma=15
    )

if __name__ == "__main__":
    main()