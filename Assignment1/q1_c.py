import numpy as np
import matplotlib.pyplot as plt

def solve_q1_c():
    x = 10.0
    w_optimal = 3.0  
    w_start = 0.0    
    n_steps = 20    
    
    # Factor is (1 - 100*eta)
    # 1. 0 < eta < 0.01
    # 2. 0.01 < eta < 0.02
    # 3. 0.02 < eta 
    etas = [0.005, 0.015, 0.023]
    regime_names = ["Monotonic Convergence", "Oscillatory Convergence", "Divergence"]
    
    # Initialize plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, eta in enumerate(etas):
        # Arrays to store history
        w_empirical = np.zeros(n_steps + 1)
        w_theoretical = np.zeros(n_steps + 1)
        
        # Initialization
        w_empirical[0] = w_start
        w_theoretical[0] = w_start
        current_w = w_start

        for t in range(n_steps):
            # Gradient calculation: dE/dw = x^2 * (w - 3)
            gradient = (x**2) * (current_w - w_optimal)
            
            # Updating the rule:
            current_w = current_w - eta * gradient
            w_empirical[t+1] = current_w
            
        # w_n = w_opt + (1 - eta*x^2)^n * (w_0 - w_opt)
        steps = np.arange(n_steps + 1)
        decay_factor = (1 - eta * (x**2))
        w_theoretical = w_optimal + (decay_factor ** steps) * (w_start - w_optimal)
        
        ax = axes[i]
        
        # Plotting empirical results as scatter points
        ax.scatter(steps, w_empirical, color='blue', label='Empirical (Simulation)', zorder=3)
        
        # Plotting theoretical prediction as a dashed line
        ax.plot(steps, w_theoretical, color='red', linestyle='--', label='Theoretical Prediction', linewidth=2)
        
        # Plotting the optimal weight line
        ax.axhline(y=w_optimal, color='green', linestyle=':', label='Optimal w (3.0)', alpha=0.7)
        
        ax.set_title(f"{regime_names[i]}\n($\\eta={eta}$)")
        ax.set_xlabel("Time Step (n)")
        ax.set_ylabel("Weight (w)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_q1_c()