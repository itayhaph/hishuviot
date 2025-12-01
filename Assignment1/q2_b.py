import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename='tagged_data.npz'):
    """
    Loads data from npz file. 
    If file doesn't exist, generates synthetic data for demonstration.
    """
    if os.path.exists(filename):
        data = np.load(filename)

        # Adjust keys if the specific file has different internal names.
        keys = list(data.keys())
        x = data[keys[0]] # taking first array as inputs
        y = data[keys[1]] # taking second array as labels
        print(f"Loaded {filename} successfully.")
    else:
        print(f"Warning: {filename} not found. Generating synthetic data for testing.")
        
    return x, y

def add_bias(x):
    """Appends a column of 1s to the input matrix x for the bias term."""
    bias = np.ones((x.shape[0], 1))
    return np.hstack((bias, x))

def solve_q2():
    filename = 'tagged_data.npz'
    X, Y = load_data(filename)
    
    
    # 1. Randomly divide into train (70%) and test (30%)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(0.7 * num_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train_raw, Y_train = X[train_idx], Y[train_idx]
    X_test_raw, Y_test = X[test_idx], Y[test_idx]
# ==========================================
    # Part B: Ridge Regression (Analytical)
    # ==========================================
    X_train = add_bias(X_train_raw)
    X_test = add_bias(X_test_raw)
    n_features = X_train.shape[1]
    print("Running Part B: Ridge Regression...",n_features)
    
    # Lambda values
    # 10^-3 to 10^5
    lambdas = [10**i for i in range(-3, 6)] 
    
    ridge_train_errors = []
    ridge_test_errors = []
    
    # Identity matrix for regularization (size of weights)
    I = np.eye(n_features)
    # Exclude bias from regularization
    # Assuming bias is at index 0 (since we used hstack((bias, x)))
    I[0, 0] = 0 
    
    for lam in lambdas:
        # Analytical solution: w* = (X^T X + lambda I)^-1 X^T Y
        # Using pseudo-inverse or solve is numerically more stable than inv
        XtX = X_train.T @ X_train
        XtY = X_train.T @ Y_train
        
        # Regularized matrix
        A = XtX + lam * I
        
        # Solve for w*
        w_star = np.linalg.solve(A, XtY)
        
        # Calculate Errors
        mse_train = np.mean((X_train @ w_star - Y_train) ** 2)/2
        mse_test = np.mean((X_test @ w_star - Y_test) ** 2)/2
        
        ridge_train_errors.append(mse_train)
        ridge_test_errors.append(mse_test)

    # Plot for Part B using subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
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
    solve_q2()