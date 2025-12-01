import numpy as np
import matplotlib.pyplot as plt

def solve_q1_f_bonus():
    mu_x = 4.0
    sigma_x = 2.0
    
    w_optimal = 3.0
    w_start = 0.0
    n_steps = 30           
    num_trials = 10000     
    
    # Selected learning rates based on analytical derivation (Convergence if eta < 0.1)
    # 1. Monotonic Convergence (0 < eta < 0.05) -> (1 - 20*eta) is positive
    # 2. Oscillatory Convergence (0.05 < eta < 0.1) -> (1 - 20*eta) is negative but > -1
    # 3. Divergence (eta > 0.1)
    etas = [0.01, 0.08, 0.11]
    regime_names = ["Monotonic Convergence", "Oscillatory Convergence", "Divergence"]
    
    # Initialize plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, eta in enumerate(etas):
        # --- Theoretical Calculation ---
        # Formula derived in Section E: E[w_n] = 3 - 3 * (1 - 20*eta)^n
        steps = np.arange(n_steps + 1)
        # Note: We use 20 because E[x^2] = Var(x) + E[x]^2 = 4 + 16 = 20
        decay_factor = (1 - 20 * eta)
        w_theoretical = w_optimal + (decay_factor ** steps) * (w_start - w_optimal)
        
        # --- Empirical Simulation (Vectorized) ---
        # We run 'num_trials' in parallel using numpy matrices
        # w_current shape: (num_trials,)
        w_current = np.full(num_trials, w_start)
        
        # To store the mean weight at each step
        w_mean_history = np.zeros(n_steps + 1)
        w_mean_history[0] = w_start
        
        for t in range(n_steps):
            # Generate stochastic inputs for all trials: X ~ N(4, 2^2)
            x_t = np.random.normal(mu_x, sigma_x, num_trials)
            
            # Calculate gradient for each trial: (w*x - 3*x) * x = x^2 * (w - 3)
            # This is the gradient of 0.5(wx - 3x)^2
            gradient = (x_t ** 2) * (w_current - w_optimal)
            
            # Update weights
            w_current = w_current - eta * gradient
            
            # Record the mean of weights across all trials
            w_mean_history[t+1] = np.mean(w_current)
            
        # --- Plotting ---
        ax = axes[i]
        
        # Plot Empirical Mean
        ax.plot(steps, w_mean_history, 'o', color='blue', label=f'Empirical Mean (Avg of {num_trials})', alpha=0.6)
        
        # Plot Theoretical Prediction
        ax.plot(steps, w_theoretical, 'r--', label='Theoretical Expectation E[w]', linewidth=2)
        
        # Plot Optimal Weight
        ax.axhline(y=w_optimal, color='green', linestyle=':', label='Optimal w (3.0)')
        
        # Formatting
        ax.set_title(f"Regime: {regime_names[i]}\n($\\eta={eta}$)")
        ax.set_xlabel("Time Step (n)")
        ax.set_ylabel("Average Weight (E[w])")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-limits for the divergent case to keep it readable
        if i == 2:
            ax.set_ylim(-10, 20)

    plt.suptitle("Stochastic Gradient Descent: Empirical Mean vs. Theoretical Prediction (Bonus Q1.f)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

if __name__ == "__main__":
    solve_q1_f_bonus()