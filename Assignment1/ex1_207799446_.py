import numpy as np
import matplotlib.pyplot as plt

def load_data(filename='tagged_data.npz'):
    data = np.load(filename)
    keys = list(data.keys())
    x = data[keys[0]] # taking first array as inputs
    y = data[keys[1]] # taking second array as labels
        
    return x, y

def add_bias(x):
    # Appending a column of 1s to the input matrix x for the bias term
    bias = np.ones((x.shape[0], 1))
    return np.hstack((bias, x))

def ex1():
    # Q1_c
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
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    plt.pause(0.1)

    # Q1_f
    mu_x = 4.0
    sigma_x = 2.0
    w_optimal = 3.0
    w_start = 0.0
    n_steps = 30           
    num_trials = 10000     
    
    # 1. Monotonic Convergence 0 < eta < 0.05
    # 2. Oscillatory Convergence 0.05 < eta < 0.1
    # 3. Divergence 0.1 < eta
    etas = [0.01, 0.08, 0.11]
    regime_names = ["Monotonic Convergence", "Oscillatory Convergence", "Divergence"]
    
    # Initialize plot
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, eta in enumerate(etas):
        # E[w_n] = 3 - 3 * (1 - 20*eta)^n
        steps = np.arange(n_steps + 1)
        decay_factor = (1 - 20 * eta)
        w_theoretical = w_optimal + (decay_factor ** steps) * (w_start - w_optimal)
        
        # We run 'num_trials' in parallel using numpy matrices
        w_current = np.full(num_trials, w_start)
        
        # storing the mean weight at each step
        w_mean_history = np.zeros(n_steps + 1)
        w_mean_history[0] = w_start
        
        for t in range(n_steps):
            # Generate inputs for all trials:
            x_t = np.random.normal(mu_x, sigma_x, num_trials)
            
            # Calculating gradient for each trial: x^2 * (w - 3)
            gradient = (x_t ** 2) * (w_current - w_optimal)
            
            # Updating weights
            w_current = w_current - eta * gradient
            
            # Recording the mean of weights across all trials
            w_mean_history[t+1] = np.mean(w_current)
            
        ax = axes[i]
        
        # Plot Empirical Mean
        ax.plot(steps, w_mean_history, 'o', color='blue', label=f'Empirical Mean (Avg of {num_trials})', alpha=0.6)
        
        # Plot Theoretical Prediction
        ax.plot(steps, w_theoretical, 'r--', label='Theoretical Expectation E[w]', linewidth=2)
        
        # Plot Optimal Weight
        ax.axhline(y=w_optimal, color='green', linestyle=':', label='Optimal w (3.0)')
        
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
    plt.pause(0.1)

    # Q2_a:
    X, Y = load_data()
    
    # Randomly divide the samples into train (70%) and test (30%)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(0.7 * num_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train_raw, Y_train = X[train_idx], Y[train_idx]
    X_test_raw, Y_test = X[test_idx], Y[test_idx]
    
    X_train = add_bias(X_train_raw)
    X_test = add_bias(X_test_raw)
    
    epochs = 100
    eta = 0.001  
    n_features = X_train.shape[1]
    
    # Initialize weights (including bias)
    w = np.random.randn(n_features) * 0.01
    train_errors = []
    test_errors = []
    
    for _ in range(epochs):
        # Shuffle training data at start of each epoch
        perm = np.random.permutation(len(Y_train))
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]
        
        # Online update: (updating after each example)
        for i in range(len(Y_train)):
            x_i = X_shuffled[i]
            y_i = Y_shuffled[i]
            
            y_hat = np.dot(w, x_i)
            gradient = (y_hat - y_i) * x_i
            
            # Update rule
            w = w - eta * gradient
        
        # Calculate errors after each epoch
        # MSE = mean((y_pred - y_true)^2)/2
        mse_train = np.mean((X_train @ w - Y_train) ** 2)/2
        mse_test = np.mean((X_test @ w - Y_test) ** 2)/2
        
        train_errors.append(mse_train)
        test_errors.append(mse_test)

    # Plot for Part A
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_errors, label='Train Error')
    plt.plot(range(1, epochs + 1), test_errors, label='Test Error', linestyle='--')
    plt.title('Part A: Online Learning Error vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)
    
    # Q2_b:
    # Lambda values from 10^-3 to 10^5
    lambdas = [10**i for i in range(-3, 6)] 
    
    ridge_train_errors = []
    ridge_test_errors = []
    
    # Identity matrix
    I = np.eye(n_features)
    # Excluding bias from regularization (at index 0)
    I[0, 0] = 0 
    
    for lam in lambdas:
        # Analytical solution: w* = (X^T X + lambda I)^-1 X^T Y
        XtX = X_train.T @ X_train
        XtY = X_train.T @ Y_train
        
        # Regularized matrix
        A = XtX + lam * I
        
        # Calculate w* using linalg.solve function
        w_star = np.linalg.solve(A, XtY)
        
        # Calculating Errors
        mse_train = np.mean((X_train @ w_star - Y_train) ** 2)/2
        mse_test = np.mean((X_test @ w_star - Y_test) ** 2)/2
        
        ridge_train_errors.append(mse_train)
        ridge_test_errors.append(mse_test)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Train Error Plot
    ax1.plot(lambdas, ridge_train_errors, 'b-o', label='Train Error')
    ax1.set_xscale('log')
    ax1.set_ylabel('MSE (Train)')
    ax1.set_title('Part B: Ridge Regression - Train Error vs Lambda')
    ax1.grid(True)
    ax1.legend()
    
    # Test Error Plot
    ax2.plot(lambdas, ridge_test_errors, 'r-s', label='Test Error')
    ax2.set_xscale('log')
    ax2.set_xlabel('Lambda (log scale)')
    ax2.set_ylabel('MSE (Test)')
    ax2.set_title('Part B: Ridge Regression - Test Error vs Lambda')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ex1()